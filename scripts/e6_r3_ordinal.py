"""E6: modifications on top of the best fine-tuned model (R3, DINOv3-B), one run per R3 seed.

    python scripts/e6_r3_ordinal.py                                   # control + ordinal, seeds 0-2
    python scripts/e6_r3_ordinal.py --variants dinov3_b_ord448         # ordinal at 448 px (about 4x slower)

Every arm starts from the same checkpoint, runs/E2_dinov3_b_fp32_s<seed>/model.pt, and gets the same recipe;
only the head (and, for dinov3_b_ord448, the input size) differs:
  1. Linear probe, then fine-tune (LP-FT): refit the head on frozen Train features (no augmentation)
     so the new head does not distort the backbone in the first steps of fine-tuning.
  2. Fine-tune everything for 12 epochs at 0.3x R3's chosen learning rate (layer decay 0.75, warm-up 1,
     patience 5), early stopping on Val-A T1 AUROC.
Arms: dinov3_b_cont (R3's softmax head, the control for the extra training), dinov3_b_ord (ordinal head),
dinov3_b_ord448 (ordinal head at 448 px). Runs are written as runs/E2_<variant>_fp32_s<seed>, so E3a, E3b,
explain.py and analyze.py treat them as quantization parents; config.yaml records init_from.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402
from timm.optim import param_groups_layer_decay  # noqa: E402

from tzq.config import CLASS_WEIGHTS, E6_VARIANTS, LLRD, MODELS, RUNS_DIR  # noqa: E402
from tzq.data import eval_transform, load_split, make_loader, train_transform  # noqa: E402
from tzq.models import build_classifier  # noqa: E402
from tzq.ordinal import OrdinalHead, ordinal_loss  # noqa: E402
from tzq.train import fit  # noqa: E402
from tzq.utils import autocast, device, finished, load_json, predict_logits, seed_all, write_preds  # noqa: E402

EPOCHS, WARMUP, PATIENCE, WD, LR_FACTOR = 12, 1, 5, 0.05, 0.3
LP_STEPS, LP_LR = 300, 1e-3


@torch.no_grad()
def train_features(model, size, mean, std, dev, workers):
    model.eval()
    ctx, _ = autocast(dev)
    dl = make_loader("train", eval_transform(size, mean, std), 64, workers=workers)
    feats = torch.zeros(len(dl.dataset), model.get_classifier().in_features)
    for x, _, idx in dl:
        with ctx:
            feats[idx] = model.forward_head(model.forward_features(x.to(dev)), pre_logits=True).float().cpu()
    return feats.to(dev)


def linear_probe(model, loss_of, size, mean, std, dev, workers):
    """Full-batch Adam on the head only, frozen backbone (the LP step of LP-FT)."""
    feats = train_features(model, size, mean, std, dev, workers)
    y = torch.tensor(load_split("train").label.to_numpy(), device=dev)
    head = model.get_classifier()
    if isinstance(head, OrdinalHead):
        head.init_thresholds(y.cpu())
    head.train()
    opt = torch.optim.Adam(head.parameters(), lr=LP_LR)
    for _ in range(LP_STEPS):
        opt.zero_grad(set_to_none=True)
        loss = loss_of(head(feats), y)
        loss.backward()
        opt.step()
    print(f"  linear probe: train loss {loss.item():.4f}", flush=True)


def run(variant, seed, workers):
    out = RUNS_DIR / f"E2_{variant}_fp32_s{seed}"
    if (out / "preds_test.csv").exists():
        print("done already:", out.name)
        return
    parent = RUNS_DIR / f"E2_{MODELS[variant]['init']}_fp32_s{seed}"
    if not finished(parent):
        print("parent not finished, skipped:", parent.name)
        return
    seed_all(seed)
    dev = device()
    model, size, mean, std = build_classifier(variant, pretrained=False)
    ordinal = isinstance(model.get_classifier(), OrdinalHead)
    state = torch.load(parent / "model.pt", map_location="cpu")
    if ordinal:
        state = {k: v for k, v in state.items() if not k.startswith("head.")}
    res = model.load_state_dict(state, strict=False)
    assert not res.unexpected_keys and all(k.startswith("head.") for k in res.missing_keys), res
    model.to(dev)

    cw = torch.tensor(CLASS_WEIGHTS, device=dev)
    ce = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
    head = model.get_classifier()
    if ordinal:
        loss_of = lambda z, y: ordinal_loss(head.cum, y, cw)  # noqa: E731
    else:
        loss_of = lambda z, y: ce(z.float(), y)  # noqa: E731

    lr = LR_FACTOR * yaml.safe_load((parent / "config.yaml").read_text())["lr"]
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "summary.json").exists():
        if not (out / "last.pt").exists():  # a resumed run restores the probed head from last.pt
            linear_probe(model, loss_of, size, mean, std, dev, workers)
        (out / "config.yaml").write_text(yaml.safe_dump(dict(
            model=variant, timm_name=MODELS[variant]["name"], init_from=parent.name, head="ordinal" if ordinal else
            "softmax", size=size, lr=lr, lr_factor=LR_FACTOR, seed=seed, epochs=EPOCHS, warmup=WARMUP,
            patience=PATIENCE, weight_decay=WD, llrd=LLRD["dinov3_b"], batch=16, accum=2, lp_steps=LP_STEPS,
            lp_lr=LP_LR, label_smoothing=0.1, class_weights=CLASS_WEIGHTS, amp=str(autocast(dev)[1]),
            gpu=torch.cuda.get_device_name() if dev.type == "cuda" else None)))
        nwd = set(getattr(model, "no_weight_decay", lambda: set())())
        groups = param_groups_layer_decay(model, WD, nwd, layer_decay=LLRD["dinov3_b"])
        for g in groups:
            g["lr"] = lr * g.pop("lr_scale", 1.0)
        opt = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999))
        train_dl = make_loader("train", train_transform(size, mean, std), 16, shuffle=True, workers=workers,
                               drop_last=True)
        val_dl = make_loader("valA", eval_transform(size, mean, std), 64, workers=workers)
        fit(model, train_dl, val_dl, out, lambda z, y, x: loss_of(z, y), opt, dev, epochs=EPOCHS,
            warmup_epochs=WARMUP, patience=PATIENCE, accum=2, resume_tag=variant)

    model.load_state_dict(torch.load(out / "model.pt", map_location=dev))
    for split in ["valA", "valB", "test"]:  # Test opened once per run, logged by write_preds
        dl = make_loader(split, eval_transform(size, mean, std), 64, workers=workers)
        write_preds(out, split, predict_logits(model, dl, dev))
    s = load_json(out / "summary.json")
    print(f"{out.name}: best Val-A AUROC {s['best_valA_auroc']:.4f} at epoch {s['best_epoch']}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["dinov3_b_cont", "dinov3_b_ord"], choices=E6_VARIANTS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    for seed in a.seeds:  # seed-major, so a partial run already gives paired control/ordinal results
        for v in a.variants:
            run(v, seed, a.workers)

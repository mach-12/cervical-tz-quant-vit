"""E4: binary models. Each subcommand runs in its own process, because both repos ship
top-level modules named `models` and `utils`.

    python scripts/e4_binary.py binatt        # B1: BinaryAttention on fine-tuned DeiT-S (R2), 3 h cap
    python scripts/e4_binary.py bhvit_fp      # B3: BHViT-Tiny 32/32, E2 recipe, LR in {1e-4, 3e-4}
    python scripts/e4_binary.py bhvit_bin     # B2: BHViT-Tiny 1/1, KD from R3, 3 h cap + stop rule
Needs external/BinaryAttention, external/BHViT and external/ckpt/bhvit_tiny/BHViT-tiny/*.pth
(see notebooks/colab_driver.ipynb).
"""
import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from timm.optim import param_groups_weight_decay  # noqa: E402

from tzq.config import CLASS_WEIGHTS, ROOT, RUNS_DIR  # noqa: E402
from tzq.cost import estimate_cost  # noqa: E402
from tzq.data import eval_transform, make_loader, train_transform  # noqa: E402
from tzq.models import build_classifier  # noqa: E402
from tzq.train import fit  # noqa: E402
from tzq.utils import device, load_json, predict_logits, save_json, seed_all, write_preds  # noqa: E402

EXT = ROOT / "external"
BHVIT_CKPT = EXT / "ckpt" / "bhvit_tiny" / "BHViT-tiny"
MEAN, STD, SIZE = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 224


def best_seed_run(model_key):
    """Best seed of an E2 parent, chosen on Val-A AUROC."""
    runs = sorted(RUNS_DIR.glob(f"E2_{model_key}_fp32_s*"))
    return max(runs, key=lambda d: load_json(d / "summary.json")["best_valA_auroc"])


def frozen_teacher(model_key, dev):
    model, *_ = build_classifier(model_key, pretrained=False)
    run = best_seed_run(model_key)
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, run.name


class LogitsOnly(nn.Module):
    """BHViT returns an ImageClassifierOutput; the rest of the pipeline expects a logits tensor."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(pixel_values=x, return_dict=True).logits


def weighted_ce():
    return nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS), label_smoothing=0.1)


def finish(model, out_dir, dev, workers, attn_bits=32):
    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=dev))
    save_json(out_dir / "cost.json", estimate_cost(model, SIZE, attn_qk_bits=attn_bits))
    for split in ["valA", "valB", "test"]:
        dl = make_loader(split, eval_transform(SIZE, MEAN, STD), 64, workers=workers)
        write_preds(out_dir, split, predict_logits(model, dl, dev))


def loaders(workers, bs=32):
    return (make_loader("train", train_transform(SIZE, MEAN, STD), bs, shuffle=True, workers=workers, drop_last=True),
            make_loader("valA", eval_transform(SIZE, MEAN, STD), 64, workers=workers))


# ----------------------------------------------------------------------------- B1

def binatt(workers):
    out = RUNS_DIR / "E4_binatt_deit_s_qk1_s0"
    if (out / "preds_test.csv").exists():
        return
    dev = device()
    seed_all(0)
    teacher, teacher_run = frozen_teacher("deit_s", dev)  # built before the repo re-registers DeiT names
    sys.path.insert(0, str(EXT / "BinaryAttention"))
    ba = importlib.import_module("models")
    model = ba.VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, num_classes=3,
                                 drop_path_rate=0.1, attn_quant=True, attn_bias=True, pv_quant=True)
    res = model.load_state_dict(teacher.state_dict(), strict=False)
    assert all("relative_position" in k for k in res.missing_keys) and not res.unexpected_keys, res
    model.to(dev)

    # Repo fine-tuning defaults (lr 5e-5 -> min 5e-6, wd 0.02, 5 warm-up epochs), hard self-distillation
    # with alpha 0.5 from the full-precision parent; protocol: 30 epochs, patience 7, our augmentation.
    lr, min_lr, wd, alpha = 5e-5, 5e-6, 0.02, 0.5
    opt = torch.optim.AdamW(param_groups_weight_decay(model, wd, model.no_weight_decay()), lr=lr)
    ce = weighted_ce().to(dev)

    def loss_fn(z, y, x):
        with torch.no_grad():
            t = teacher(x).float()
        return (1 - alpha) * ce(z, y) + alpha * F.cross_entropy(z, t.argmax(1))

    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(dict(
        model="BinaryAttention DeiT-S", init=teacher_run, parent=teacher_run, teacher=teacher_run, attn_quant=True, attn_bias=True,
        pv_quant=True, lr=lr, min_lr=min_lr, weight_decay=wd, distillation="hard", alpha=alpha, epochs=30,
        patience=7, warmup=5, time_cap_h=3)))
    train_dl, val_dl = loaders(workers)
    fit(model, train_dl, val_dl, out, loss_fn, opt, dev, epochs=30, warmup_epochs=5, patience=7,
        min_lr_ratio=min_lr / lr, time_cap_h=3)
    finish(model, out, dev, workers, attn_bits=1)


# ----------------------------------------------------------------------------- BHViT

def bhvit(bits, some_fp, ckpt, dev):
    sys.path.insert(0, str(EXT / "BHViT"))
    torch.serialization.add_safe_globals([argparse.Namespace])
    ck = torch.load(BHVIT_CKPT / ckpt, map_location="cpu", weights_only=True)
    args = ck["args"]
    args.nb_classes, args.some_fp, args.drop_path = 1000, some_fp, 0.0
    get_model = importlib.import_module("models").get_model
    net = get_model(args, str(EXT / "BHViT" / "configs" / "BHViT_tiny"), "dbhvit", bits, bits)
    res = net.load_state_dict(ck["model"], strict=False)
    net.classifier = nn.Linear(net.classifier.in_features, 3)
    return LogitsOnly(net).to(dev), dict(missing=res.missing_keys, unexpected=res.unexpected_keys)


def bhvit_fp(workers, lrs=(1e-4, 3e-4)):
    """B3. The repo publishes no 32-bit BHViT; this is the BHViT-Tiny architecture at 32/32,
    initialised from the latent weights of the "somefp 66.0" checkpoint, fine-tuned with the E2 recipe."""
    final = RUNS_DIR / "E4_bhvit_tiny_fp_fp32_s0"
    if (final / "preds_test.csv").exists():
        return
    dev = device()
    scores = {}
    for lr in lrs:
        out = RUNS_DIR / "lrsearch" / f"E4_bhvit_tiny_fp_fp32_s0_lr{lr:g}"
        if (out / "summary.json").exists():
            scores[lr] = load_json(out / "summary.json")["best_valA_auroc"]
            continue
        seed_all(0)
        model, load_info = bhvit(32, True, "best66.0r.pth", dev)
        opt = torch.optim.AdamW(param_groups_weight_decay(model, 0.05), lr=lr, betas=(0.9, 0.999))
        ce = weighted_ce().to(dev)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.yaml").write_text(yaml.safe_dump(dict(model="BHViT-Tiny 32/32", init="best66.0r.pth",
                                                             lr=lr, load=load_info, recipe="E2")))
        train_dl, val_dl = loaders(workers)
        scores[lr] = fit(model, train_dl, val_dl, out, lambda z, y, x: ce(z, y), opt, dev,
                         epochs=30, warmup_epochs=2, patience=7)["best_valA_auroc"]
    chosen = max(scores, key=scores.get)
    save_json(RUNS_DIR / "lrsearch" / "E4_bhvit_tiny_fp_selected.json", dict(scores=scores, chosen=chosen))
    src = RUNS_DIR / "lrsearch" / f"E4_bhvit_tiny_fp_fp32_s0_lr{chosen:g}"
    final.mkdir(parents=True, exist_ok=True)
    for f in ("model.pt", "config.yaml", "summary.json"):
        (final / f).write_bytes((src / f).read_bytes())
    model, _ = bhvit(32, True, "best66.0r.pth", dev)
    finish(model, final, dev, workers)


def bhvit_bin(workers, lr=1e-4):
    """B2: 0.5 weighted CE + 0.5 T^2 KL to R3 (T = 4) + the repo's 0.1 x oscillation regularisation."""
    out = RUNS_DIR / "E4_bhvit_tiny_bin_bin_s0"
    if (out / "preds_test.csv").exists():
        return
    dev = device()
    teacher, teacher_run = frozen_teacher("dinov3_b", dev)
    T, reg_w = 4.0, 0.1
    ce = weighted_ce().to(dev)
    reg_keys = ("cov1.weight", "cov2.weight", "cov3.weight", "dense.weight", "query.weight", "key.weight",
                "value.weight")
    for attempt, cur_lr in enumerate([lr, 0.3 * lr]):
        seed_all(0)
        model, load_info = bhvit(1, False, "best64.0.pth", dev)
        reg_params = [p for n, p in model.named_parameters() if any(k in n for k in reg_keys)]

        def loss_fn(z, y, x):
            with torch.no_grad():
                t = teacher(x).float()
            kd = F.kl_div(F.log_softmax(z / T, 1), F.log_softmax(t / T, 1), reduction="batchmean",
                          log_target=True) * T * T
            reg = torch.stack([(p.abs() - 1).abs().mean() for p in reg_params]).mean()
            return 0.5 * ce(z, y) + 0.5 * kd + reg_w * reg

        # Repo settings: AdamW, no weight decay, cosine, no warm-up.
        opt = torch.optim.AdamW(model.parameters(), lr=cur_lr, weight_decay=0.0)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.yaml").write_text(yaml.safe_dump(dict(
            model="BHViT-Tiny 1/1", init="best64.0.pth", parent="E4_bhvit_tiny_fp_fp32_s0", teacher=teacher_run, kd_T=T, reg_weight=reg_w, lr=cur_lr,
            attempt=attempt, epochs=100, patience=15, time_cap_h=3, load=load_info)))
        train_dl, val_dl = loaders(workers)
        s = fit(model, train_dl, val_dl, out, loss_fn, opt, dev, epochs=100, warmup_epochs=0, patience=15,
                time_cap_h=3, abort_rule=(10, 0.60))
        if s["status"] != "aborted":
            finish(model, out, dev, workers)
            return
        print(f"B2 stop rule hit at lr {cur_lr:g}")
    (out / "FAILED.txt").write_text("Val-A T1 AUROC < 0.60 after 10 epochs at lr and 0.3x lr; reported as a failure.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["binatt", "bhvit_fp", "bhvit_bin"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    {"binatt": binatt, "bhvit_fp": bhvit_fp, "bhvit_bin": bhvit_bin}[a.which](a.workers)

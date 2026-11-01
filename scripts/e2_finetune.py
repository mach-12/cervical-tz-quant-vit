"""E2: fine-tune R1-R5 with the shared recipe.

Full protocol for one model (LR search on seed 0, then seeds 1-2 at the chosen LR, then Test once):
    python scripts/e2_finetune.py --model deit_s
Single run (no Test):
    python scripts/e2_finetune.py --model deit_s --lr 3e-5 --seed 0
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402
from timm.optim import param_groups_layer_decay, param_groups_weight_decay  # noqa: E402

from tzq.config import CLASS_WEIGHTS, LLRD, LR_GRID, MODELS, RUNS_DIR  # noqa: E402
from tzq.data import eval_transform, make_loader, train_transform  # noqa: E402
from tzq.models import build_classifier  # noqa: E402
from tzq.train import fit  # noqa: E402
from tzq.utils import (autocast, device, load_json, predict_logits, save_json,  # noqa: E402
                       seed_all, write_preds)

EPOCHS, WARMUP, PATIENCE, WD = 30, 2, 7, 0.05


def run_id(model, seed):
    return f"E2_{model}_fp32_s{seed}"


def search_dir(model, lr, seed):
    return RUNS_DIR / "lrsearch" / f"{run_id(model, seed)}_lr{lr:g}"


def train_one(model_key, lr, seed, out_dir: Path, epochs=EPOCHS, limit=None, workers=4):
    if (out_dir / "preds_valB.csv").exists():
        print("done already:", out_dir)
        return load_json(out_dir / "summary.json")
    seed_all(seed)
    dev = device()
    model, size, mean, std = build_classifier(model_key)
    model.to(dev)
    big = model_key == "dinov3_b"
    bs, accum = (16, 2) if big else (32, 1)

    train_dl = make_loader("train", train_transform(size, mean, std), bs, shuffle=True,
                           workers=workers, limit=limit, drop_last=True)
    val_dl = make_loader("valA", eval_transform(size, mean, std), 64, workers=workers, limit=limit)

    nwd = set(getattr(model, "no_weight_decay", lambda: set())())
    if model_key in LLRD:
        groups = param_groups_layer_decay(model, WD, nwd, layer_decay=LLRD[model_key])
    else:
        groups = param_groups_weight_decay(model, WD, nwd)
    for g in groups:
        g["lr"] = lr * g.pop("lr_scale", 1.0)
    opt = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999))
    crit = nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS, device=dev), label_smoothing=0.1)

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(model=model_key, timm_name=MODELS[model_key]["name"], lr=lr, seed=seed, epochs=epochs,
               warmup=WARMUP, patience=PATIENCE, weight_decay=WD, batch=bs, accum=accum, size=size,
               llrd=LLRD.get(model_key), label_smoothing=0.1, class_weights=CLASS_WEIGHTS,
               amp=str(autocast(dev)[1]), device=str(dev), limit=limit,
               gpu=torch.cuda.get_device_name() if dev.type == "cuda" else None)
    if (out_dir / "summary.json").exists():  # trained in an earlier session; only predictions are missing
        summary = load_json(out_dir / "summary.json")
    else:
        (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
        summary = fit(model, train_dl, val_dl, out_dir, lambda z, y, x: crit(z, y), opt, dev,
                      epochs=epochs, warmup_epochs=WARMUP, patience=PATIENCE, accum=accum, resume_tag=f"lr{lr:g}")

    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=dev))
    for split in ["valA", "valB"]:
        dl = make_loader(split, eval_transform(size, mean, std), 64, workers=workers)
        write_preds(out_dir, split, predict_logits(model, dl, dev))
    return summary


def predict_test(model_key, run_dir: Path, workers=4):
    if (run_dir / "preds_test.csv").exists():
        return
    dev = device()
    model, size, mean, std = build_classifier(model_key, pretrained=False)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    model.to(dev)
    dl = make_loader("test", eval_transform(size, mean, std), 64, workers=workers)
    write_preds(run_dir, "test", predict_logits(model, dl, dev))
    print("Test opened for", run_dir.name)


def full_protocol(model_key, seeds, workers):
    grid = LR_GRID[model_key]
    if len(grid) == 1:
        chosen = grid[0]
        train_one(model_key, chosen, 0, RUNS_DIR / run_id(model_key, 0), workers=workers)
    else:
        scores = {lr: train_one(model_key, lr, 0, search_dir(model_key, lr, 0), workers=workers)["best_valA_auroc"]
                  for lr in grid}
        chosen = max(scores, key=scores.get)
        save_json(RUNS_DIR / "lrsearch" / f"E2_{model_key}_selected.json", dict(scores=scores, chosen=chosen))
        final0 = RUNS_DIR / run_id(model_key, 0)
        if not final0.exists():
            shutil.copytree(search_dir(model_key, chosen, 0), final0)
    print(f"{model_key}: chosen lr {chosen:g}")
    for s in seeds:
        if s != 0:
            train_one(model_key, chosen, s, RUNS_DIR / run_id(model_key, s), workers=workers)
    for s in seeds:
        predict_test(model_key, RUNS_DIR / run_id(model_key, s), workers=workers)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--limit", type=int, help="smoke test: images per split")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.lr is not None:
        out = a.out or search_dir(a.model, a.lr, a.seed)
        print(train_one(a.model, a.lr, a.seed, out, a.epochs, a.limit, a.workers)["best_valA_auroc"])
    else:
        seeds = a.seeds or ([0] if a.model == "resnet50" else [0, 1, 2])
        full_protocol(a.model, seeds, a.workers)

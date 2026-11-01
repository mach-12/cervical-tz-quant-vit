"""E4: binarized attention (B1, BinaryAttention on the fine-tuned DeiT-S parent).

    python scripts/e4_binary.py binatt        # 3 h cap
Needs external/BinaryAttention (see notebooks/colab_driver.ipynb). BHViT (B2, B3) was removed from the study.
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
    if not (out / "summary.json").exists():
        train_dl, val_dl = loaders(workers)
        fit(model, train_dl, val_dl, out, loss_fn, opt, dev, epochs=30, warmup_epochs=5, patience=7,
            min_lr_ratio=min_lr / lr, time_cap_h=3, resume_tag=f"lr{lr:g}")
    finish(model, out, dev, workers, attn_bits=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["binatt"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    {"binatt": binatt}[a.which](a.workers)

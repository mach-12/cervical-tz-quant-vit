"""E3a: weight-only sweep at 8, 4, 2, 1 bits for the parents R2-R5, every seed.

    python scripts/e3a_weight_sweep.py --models deit_s mnv4_m tinyvit_11m dinov3_b
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from tzq.config import PARENTS, QUANT_DIR, RUNS_DIR  # noqa: E402
from tzq.data import eval_transform, make_loader  # noqa: E402
from tzq.metrics import t1_auroc, tz3_score  # noqa: E402
from tzq.models import build_classifier  # noqa: E402
from tzq.quant import quantize_model  # noqa: E402
from tzq.utils import device, predict_logits, save_json, write_preds  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=PARENTS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--bits", type=int, nargs="+", default=[8, 4, 2, 1])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    dev = device()
    for key in a.models:
        for seed in a.seeds:
            parent = RUNS_DIR / f"E2_{key}_fp32_s{seed}"
            if not (parent / "model.pt").exists():
                print("missing parent", parent)
                continue
            state = torch.load(parent / "model.pt", map_location="cpu")
            for bits in a.bits:
                out = QUANT_DIR / f"E3a_{key}_w{bits}_s{seed}"
                if (out / "preds_test.csv").exists():
                    continue
                model, size, mean, std = build_classifier(key, pretrained=False)
                model.load_state_dict(state)
                cost = quantize_model(model, bits)
                model.to(dev)
                for split in ["valA", "valB", "test"]:
                    dl = make_loader(split, eval_transform(size, mean, std), 64, workers=a.workers)
                    z = predict_logits(model, dl, dev)
                    write_preds(out, split, z)
                    if split == "valA":
                        cost["valA_auroc"] = t1_auroc(dl.dataset.df.label, tz3_score(z))
                save_json(out / "cost.json", dict(parent=parent.name, **cost))
                print(f"{out.name}: valA AUROC {cost['valA_auroc']:.4f}, {cost['size_mb_simulated']:.2f} MB", flush=True)

"""E1: logistic-regression probes on frozen embeddings (P1-P3) and HSV histograms (C1).

    python scripts/e1_probe.py --probe hsv dinov2_b dinov3_b medsiglip
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from tzq.config import CLASS_WEIGHTS, FEATURES_DIR, RUNS_DIR, SPLITS  # noqa: E402
from tzq.data import RAW_DIR, eval_transform, load_split, make_loader  # noqa: E402
from tzq.metrics import t1_auroc, tz3_score  # noqa: E402
from tzq.models import FeatureExtractor  # noqa: E402
from tzq.utils import autocast, device, save_json, write_preds  # noqa: E402

C_GRID = np.logspace(-4, 2, 13)


def hsv_hist(path, bins=32):
    img = Image.open(path).convert("RGB")
    img.thumbnail((256, 256))  # histogram of the unpadded crop
    hsv = np.asarray(img.convert("HSV")).reshape(-1, 3)
    h = np.concatenate([np.histogram(hsv[:, c], bins=bins, range=(0, 256))[0] for c in range(3)])
    return (h / h.sum()).astype(np.float32)


@torch.no_grad()
def extract(probe, workers):
    out = FEATURES_DIR / probe
    if all((out / f"{s}.npy").exists() for s in SPLITS):
        return
    out.mkdir(parents=True, exist_ok=True)
    if probe == "hsv":
        for s in SPLITS:
            np.save(out / f"{s}.npy", np.stack([hsv_hist(RAW_DIR / p) for p in load_split(s).path]))
        return
    dev = device()
    fx = FeatureExtractor(probe).to(dev)
    ctx, _ = autocast(dev)
    tf = eval_transform(fx.size, fx.mean, fx.std)
    for s in SPLITS:
        dl = make_loader(s, tf, 32, workers=workers)
        chunks = {}
        for x, _, idx in dl:
            with ctx:
                f = fx(x.to(dev))
            for i, v in zip(idx.tolist(), f.float().cpu().numpy()):
                chunks[i] = v
        feats = np.stack([chunks[i] for i in range(len(dl.dataset))]).astype(np.float32)
        np.save(out / f"{s}.npy", feats)
        print(probe, s, feats.shape, flush=True)


def fit_probe(probe):
    X = {s: np.load(FEATURES_DIR / probe / f"{s}.npy") for s in SPLITS}
    y = {s: load_split(s).label.to_numpy() for s in SPLITS}
    sc = StandardScaler().fit(X["train"])
    X = {s: sc.transform(v) for s, v in X.items()}
    cw = dict(enumerate(CLASS_WEIGHTS))
    results = []
    for C in C_GRID:
        clf = LogisticRegression(solver="lbfgs", max_iter=5000, class_weight=cw, C=C).fit(X["train"], y["train"])
        z = clf.decision_function(X["valA"])
        results.append(dict(C=float(C), auroc=t1_auroc(y["valA"], tz3_score(z)),
                            logloss=log_loss(y["valA"], clf.predict_proba(X["valA"]), labels=[0, 1, 2]), clf=clf))
        print(f"{probe} C={C:.0e} valA AUROC {results[-1]['auroc']:.4f} logloss {results[-1]['logloss']:.4f}")
    best = max(results, key=lambda r: (round(r["auroc"], 6), -r["logloss"]))
    run = RUNS_DIR / f"E1_{probe}_fp32_s0"
    for s in ["valA", "valB", "test"]:
        write_preds(run, s, best["clf"].decision_function(X[s]))
    with open(run / "probe.pkl", "wb") as f:
        pickle.dump(dict(scaler=sc, clf=best["clf"]), f)
    save_json(run / "summary.json", dict(chosen_C=best["C"], best_valA_auroc=best["auroc"],
                                         grid=[{k: r[k] for k in ("C", "auroc", "logloss")} for r in results]))
    print(f"{probe}: chosen C={best['C']:.0e}, valA AUROC {best['auroc']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", nargs="+", default=["hsv", "dinov2_b", "dinov3_b", "medsiglip"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    for p in a.probe:
        extract(p, a.workers)
        fit_probe(p)

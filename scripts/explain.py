"""Post-hoc explainability for the trained E2 models and their E3a weight-quantized variants.

    python scripts/explain.py                                 # every finished E2 run, Val-B
    python scripts/explain.py --models deit_s --seeds 0 --bits 8 4

For each model (and, for the quantization parents, each simulated bit width rebuilt from the parent's
model.pt exactly as in E3a):
  - Grad-CAM of log p(TZ3), the score behind the referral decision. CNNs and TinyViT use the last feature
    map; DeiT-S and DINOv3-B the patch tokens entering the last block (their prefix tokens are dropped).
  - Deletion faithfulness: 16 px blocks are blanked (set to the normalisation mean) from the most to the
    least salient, and p(TZ3) is recorded; the same with a random block order as the baseline.
  - Stability under quantization: per-image Spearman correlation and top-20% IoU between the quantized
    and the full-precision map, and agreement of the predicted class.
Outputs go to explain/ next to runs/ (on Google Drive in Colab):
  maps/<model>_<fp32|wN>_s<seed>_<split>.npz   maps, logits, deletion curves (finished files are skipped)
  summary.csv, table_explain.md, fig_stability.png/.pdf
  panels/cases_<model>.png      seed 0, full precision: detected TZ3, missed TZ3, false alarms
  panels/models.png             the same images across all models (full precision, seed 0)
  panels/quant_<model>.png      the same images across bit widths (seed 0)
  paper_gradcam.pdf, table_explain.tex   compact versions for paper.tex
E4 binary models are not covered (their code lives in the third-party repos).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from tzq.config import E6_VARIANTS, EXPLAIN_DIR, MODELS, QUANT_PARENTS as PARENTS, RUNS_DIR, TZ3  # noqa: E402
from tzq.data import CACHE_DIR, eval_transform, load_split, make_loader  # noqa: E402
from tzq.metrics import fit_temperature, spec_threshold, t1_auroc, tz3_score  # noqa: E402
from tzq.models import build_classifier  # noqa: E402
from tzq.quant import quantize_model  # noqa: E402
from tzq.utils import device, finished, log_test_access, logits_of, read_preds, stale  # noqa: E402

FRACS = np.linspace(0, 1, 11)  # share of blocks deleted
BLOCK = 16
TOP = 0.20  # top share of map cells for the IoU
ORDER = list(MODELS)
BIT_ORDER = ["fp32", "w8", "w4", "w2", "w1"]
NAMES = {"resnet50": "R1 ResNet-50", "deit_s": "R2 DeiT-S", "dinov3_b": "R3 DINOv3-B",
         "mnv4_m": "R4 MobileNetV4-M", "tinyvit_11m": "R5 TinyViT-11M", "dinov3_b_cont": "R3-C DINOv3-B cont.",
         "dinov3_b_ord": "R3-O DINOv3-B ordinal", "dinov3_b_ord448": "R3-O448 DINOv3-B ordinal 448"}
PAPER_FP = ["resnet50", "deit_s", "dinov3_b", "mnv4_m", "tinyvit_11m"]  # columns of the paper figure


def gradcam(model, x):
    """Grad-CAM of log p(TZ3). Returns (maps (B, h, w) scaled to [0, 1] per image, logits (B, 3))."""
    prefix = getattr(model, "num_prefix_tokens", None) if hasattr(model, "blocks") else None
    store = {}
    x = x.requires_grad_(True)  # parameters are frozen, so this is what builds the graph
    if prefix is not None:
        def hook(m, inp, out):
            store["a"] = out
            out.register_hook(lambda g: store.__setitem__("g", g))
        h = model.blocks[-1].norm1.register_forward_hook(hook)
        z = model(x)
        h.remove()
    else:
        a = model.forward_features(x)
        a.register_hook(lambda g: store.__setitem__("g", g))
        store["a"] = a
        z = model.forward_head(a)
    torch.log_softmax(z.float(), 1)[:, TZ3].sum().backward()  # images are independent in eval mode
    a, g = store["a"].detach().float(), store["g"].float()
    if prefix is not None:
        a, g = a[:, prefix:], g[:, prefix:]
        n = int(round(a.shape[1] ** 0.5))
        cam = (g.mean(1, keepdim=True) * a).sum(2).reshape(-1, n, n)
    else:
        cam = (g.mean((2, 3), keepdim=True) * a).sum(1)
    cam = cam.clamp_min(0)
    return cam / cam.flatten(1).amax(1).clamp_min(1e-8)[:, None, None], z.detach().float()


@torch.no_grad()
def deletion(model, x, cam, gen=None):
    """p(TZ3) as BLOCK px blocks are blanked in order of saliency (or at random when gen is given)."""
    S = x.shape[-1]
    g = S // BLOCK
    if gen is None:
        score = F.interpolate(cam[:, None], size=(g, g), mode="bilinear", align_corners=False).flatten(1)
    else:
        score = torch.rand(len(x), g * g, generator=gen).to(x.device)
    rank = score.argsort(1, descending=True).argsort(1)
    out = []
    for f in FRACS:
        keep = (rank >= round(f * g * g)).float().reshape(-1, 1, g, g)
        m = F.interpolate(keep, size=(S, S), mode="nearest")
        out.append(torch.softmax(model(x * m).float(), 1)[:, TZ3])
    return torch.stack(out, 1)


def explain_variant(model, size, mean, std, split, dev, batch, workers):
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    gen = torch.Generator().manual_seed(0)
    dl = make_loader(split, eval_transform(size, mean, std), batch, workers=workers)
    cams, logits, dels, rands = [], [], [], []
    for x, _, _ in dl:
        x = x.to(dev)
        cam, z = gradcam(model, x)
        x = x.detach()
        cams.append(cam.cpu())
        logits.append(z.cpu())
        dels.append(deletion(model, x, cam).cpu())
        rands.append(deletion(model, x, cam, gen).cpu())
    cat = lambda v: torch.cat(v).numpy()  # noqa: E731
    df = dl.dataset.df
    return dict(image_id=df.image_id.to_numpy().astype(str), label=df.label.to_numpy(), cam=cat(cams).astype(np.float16),
                logits=cat(logits), del_curve=cat(dels), rand_curve=cat(rands), size=size)


def compute(a, dev):
    maps = EXPLAIN_DIR / "maps"
    maps.mkdir(parents=True, exist_ok=True)
    for key in a.models:
        for seed in a.seeds:
            parent = RUNS_DIR / f"E2_{key}_fp32_s{seed}"
            if not finished(parent):
                print("parent not finished, skipped:", parent.name)
                continue
            state = torch.load(parent / "model.pt", map_location="cpu")
            for bits in [32] + (a.bits if key in PARENTS else []):
                tag = "fp32" if bits == 32 else f"w{bits}"
                out = maps / f"{key}_{tag}_s{seed}_{a.split}.npz"
                if out.exists() and not stale(out, parent):
                    continue
                if a.split == "test":
                    log_test_access(f"explain_{out.stem}")
                model, size, mean, std = build_classifier(key, pretrained=False)
                model.load_state_dict(state)
                if bits < 32:
                    quantize_model(model, bits)
                res = explain_variant(model, size, mean, std, a.split, dev, a.batch, a.workers)
                tmp = out.with_name(out.stem + ".tmp.npz")
                np.savez_compressed(tmp, **res)
                tmp.replace(out)
                print("wrote", out.name, flush=True)


# ---------------------------------------------------------------- summary and figures

def auc(curves):
    return ((curves[:, 1:] + curves[:, :-1]) / 2 * np.diff(FRACS)).sum(1)


def rank_corr(a, b):
    ra = a.argsort(1).argsort(1).astype(float)
    rb = b.argsort(1).argsort(1).astype(float)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1)).clip(1e-12)


def top_iou(a, b):
    k = max(1, round(TOP * a.shape[1]))
    ta = np.zeros_like(a, bool)
    tb = np.zeros_like(b, bool)
    np.put_along_axis(ta, np.argsort(-a, 1)[:, :k], True, 1)
    np.put_along_axis(tb, np.argsort(-b, 1)[:, :k], True, 1)
    return (ta & tb).sum(1) / (ta | tb).sum(1)


def load_all(split):
    res = {}
    for f in sorted((EXPLAIN_DIR / "maps").glob(f"*_s*_{split}.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        stem = f.stem[: -len(split) - 1]
        key, tag, seed = stem.rsplit("_", 2)
        res[(key, tag, int(seed[1:]))] = dict(np.load(f))
    return res


def summarize(res):
    rows = []
    for (key, tag, seed), r in res.items():
        tz3 = r["label"] == TZ3
        d, rnd = auc(r["del_curve"][tz3]), auc(r["rand_curve"][tz3])
        row = dict(model=key, bits=tag, seed=seed, n=len(r["label"]), n_tz3=int(tz3.sum()),
                   deletion_auc=d.mean(), random_auc=rnd.mean(), deletion_ratio=d.mean() / rnd.mean(),
                   p_tz3_full=r["del_curve"][tz3, 0].mean())
        fp = res.get((key, "fp32", seed))
        if tag != "fp32" and fp is not None:
            a, b = fp["cam"].reshape(len(tz3), -1).astype(float), r["cam"].reshape(len(tz3), -1).astype(float)
            row.update(spearman_vs_fp32=np.nanmean(rank_corr(a, b)), iou20_vs_fp32=top_iou(a, b).mean(),
                       pred_agree_vs_fp32=(fp["logits"].argmax(1) == r["logits"].argmax(1)).mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    df["o1"] = df.model.map(ORDER.index)
    df["o2"] = df.bits.map(BIT_ORDER.index)
    return df.sort_values(["o1", "o2", "seed"]).drop(columns=["o1", "o2"]).reset_index(drop=True)


def write_table(df, path):
    cols = ["deletion_auc", "random_auc", "deletion_ratio", "spearman_vs_fp32", "iou20_vs_fp32", "pred_agree_vs_fp32"]
    cols = [c for c in cols if c in df]
    g = df.groupby(["model", "bits"], sort=False)
    lines = ["| Model | Weights | Seeds | " + " | ".join(cols) + " |", "|" + " --- |" * (len(cols) + 3)]
    for (key, tag), d in g:
        cells = []
        for c in cols:
            v = d[c].dropna()
            cells.append("" if v.empty else f"{v.mean():.3f}" + (f" ± {v.std():.3f}" if len(v) > 1 else ""))
        lines.append(f"| {NAMES.get(key, key)} | {tag} | {len(d)} | " + " | ".join(cells) + " |")
    lines += ["", "Deletion AUC: mean p(TZ3) over the deletion curve on true-TZ3 images (lower = more faithful); "
              "ratio = deletion / random-order AUC. Stability columns compare each quantized map with its "
              "full-precision parent (same seed)."]
    path.write_text("\n".join(lines) + "\n")


def fig_stability(df):
    q = df[df.model.isin(PARENTS)]
    if q.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    x = np.arange(len(BIT_ORDER))
    for key in [k for k in ORDER if k in set(q.model)]:
        d = q[q.model == key].groupby("bits")
        for ax, col in zip(axes, ["spearman_vs_fp32", "deletion_ratio"]):
            m, s = d[col].mean(), d[col].std()
            tags = [t for t in BIT_ORDER if t in m.index and not np.isnan(m[t])]
            ax.errorbar([BIT_ORDER.index(t) for t in tags], [m[t] for t in tags],
                        yerr=[0 if np.isnan(s[t]) else s[t] for t in tags], marker="o", capsize=3,
                        label=NAMES[key])
    for ax, lab in zip(axes, ["Grad-CAM Spearman vs. FP32", "Deletion AUC / random (lower = faithful)"]):
        ax.set_xticks(x, ["32", "8", "4", "2", "1"])
        ax.set_xlabel("Weight bits")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[1].axhline(1, color="grey", lw=0.8, ls="--")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(EXPLAIN_DIR / f"fig_stability.{ext}", dpi=200)
    plt.close(fig)


def show(split, ids, size):
    paths = load_split(split).set_index("image_id").path
    return [np.asarray(Image.open(CACHE_DIR / paths[i]).convert("RGB").resize((size, size), Image.BICUBIC))
            for i in ids]


def overlay(ax, img, cam, title):
    up = F.interpolate(torch.tensor(cam, dtype=torch.float32)[None, None], size=img.shape[:2],
                       mode="bilinear", align_corners=False)[0, 0].numpy()
    ax.imshow(img)
    ax.imshow(up, cmap="jet", alpha=0.4, vmin=0, vmax=1)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def grid(rows, cols, path, header):
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(1.7 * len(cols), 1.8 * len(rows)), squeeze=False)
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            ax = axes[i, j]
            if cell is None:
                ax.axis("off")
            elif cell[1] is None:
                ax.imshow(cell[0])
                ax.set_title(cell[2], fontsize=7)
                ax.axis("off")
            else:
                overlay(ax, *cell)
    for j, c in enumerate(cols):
        axes[0, j].set_title(c + "\n" + axes[0, j].get_title(), fontsize=7)
    fig.suptitle(header, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cut_and_T(key, seed=0):
    """Val-A temperature and the specificity-0.80 cut of the full-precision run, as in the paper."""
    pa = read_preds(RUNS_DIR / f"E2_{key}_fp32_s{seed}", "valA")
    z, y = logits_of(pa), pa.label.to_numpy()
    T = fit_temperature(z, y)
    return T, spec_threshold(y, tz3_score(z, T))


def panels(res, split, n=4):
    out = EXPLAIN_DIR / "panels"
    out.mkdir(exist_ok=True)
    labels = {0: "TZ1", 1: "TZ2", 2: "TZ3"}
    for key in [k for k in ORDER if (k, "fp32", 0) in res]:
        r = res[(key, "fp32", 0)]
        T, cut = cut_and_T(key)
        s = tz3_score(r["logits"], T)
        y = r["label"]
        idx = np.arange(len(y))
        groups = {"detected TZ3": idx[(y == TZ3) & (s > cut)][np.argsort(-s[(y == TZ3) & (s > cut)])][:n],
                  "missed TZ3": idx[(y == TZ3) & (s <= cut)][np.argsort(s[(y == TZ3) & (s <= cut)])][:n],
                  "false alarm": idx[(y != TZ3) & (s > cut)][np.argsort(-s[(y != TZ3) & (s > cut)])][:n]}
        rows = []
        for name, ii in groups.items():
            if not len(ii):
                continue
            imgs = show(split, r["image_id"][ii], int(r["size"]))
            row = [(img, r["cam"][i], f"{name}: {labels[y[i]]}, s={s[i]:.2f}") for img, i in zip(imgs, ii)]
            rows.append(row + [None] * (n - len(row)))
        if rows:
                grid(rows, [""] * n, out / f"cases_{key}.png",
                 f"{NAMES[key]} (seed 0, FP32), {split}: Grad-CAM of p(TZ3); cut at Val-A specificity 0.80 = {cut:.2f}")

    # A fixed, model-independent set: 3 images per class, drawn with seed 0.
    first = next(iter(res.values()))
    rng = np.random.default_rng(0)
    pick = np.concatenate([rng.choice(np.flatnonzero(first["label"] == c), 3, replace=False) for c in (2, 1, 0)])
    ids = first["image_id"][pick]
    fp = [k for k in ORDER if (k, "fp32", 0) in res]
    if fp:
        imgs = show(split, ids, 224)
        rows = []
        for img, i, iid in zip(imgs, pick, ids):
            row = [(img, None, labels[first["label"][i]])]
            for key in fp:
                r = res[(key, "fp32", 0)]
                j = int(np.flatnonzero(r["image_id"] == iid)[0])
                row.append((img, r["cam"][j], f"p={tz3_score(r['logits'][j:j + 1])[0]:.2f}"))
            rows.append(row)
        grid(rows, ["image"] + [NAMES[k] for k in fp], out / "models.png",
             f"Grad-CAM of p(TZ3), FP32, seed 0 ({split}; p = untempered softmax)")
    for key in [k for k in PARENTS if (k, "fp32", 0) in res]:
        tags = [t for t in BIT_ORDER if (key, t, 0) in res]
        if len(tags) < 2:
            continue
        imgs = show(split, ids, int(res[(key, "fp32", 0)]["size"]))
        rows = []
        for img, i, iid in zip(imgs, pick, ids):
            row = [(img, None, labels[first["label"][i]])]
            for t in tags:
                r = res[(key, t, 0)]
                j = int(np.flatnonzero(r["image_id"] == iid)[0])
                row.append((img, r["cam"][j], f"p={tz3_score(r['logits'][j:j + 1])[0]:.2f}"))
            rows.append(row)
        grid(rows, ["image"] + tags, out / f"quant_{key}.png",
             f"{NAMES[key]}, seed 0: Grad-CAM of p(TZ3) across weight bit widths ({split})")


def paper_figure(res, split):
    """One full-width figure: 3 images x (R1-R5 at FP32, then the best parent at 8/4/2/1 bits), seed 0.

    Images, scored by the best parent (Val-A AUROC): its highest- and lowest-scoring TZ3 (labelled
    detected or missed at its specificity-0.80 cut) and its lowest-scoring TZ1/2.
    """
    fp = [k for k in PAPER_FP if (k, "fp32", 0) in res]
    parents = [k for k in PARENTS if (k, "fp32", 0) in res]
    if not parents:
        return
    va = {}
    for k in parents:
        pa = read_preds(RUNS_DIR / f"E2_{k}_fp32_s0", "valA")
        va[k] = t1_auroc(pa.label, tz3_score(logits_of(pa)))
    best = max(va, key=va.get)
    r = res[(best, "fp32", 0)]
    T, cut = cut_and_T(best)
    s, y = tz3_score(r["logits"], T), r["label"]
    tz, neg = np.flatnonzero(y == TZ3), np.flatnonzero(y != TZ3)
    picks = [(i, f"TZ3, {'detected' if s[i] > cut else 'missed'}") for i in (tz[np.argmax(s[tz])], tz[np.argmin(s[tz])])]
    i = neg[np.argmin(s[neg])]
    picks = [(n, i) for i, n in picks] + [(f"TZ{y[i] + 1}", i)]
    cols = [(k, "fp32") for k in fp] + [(best, t) for t in BIT_ORDER[1:] if (best, t, 0) in res]
    heads = [NAMES[k].split()[0] + (" FP32" if t == "fp32" else f" W{t[1:]}") for k, t in cols]
    fig, axes = plt.subplots(len(picks), len(cols) + 1, figsize=(7.16, 0.68 * len(picks) + 0.25), squeeze=False)
    for i, (name, ix) in enumerate(picks):
        iid = r["image_id"][ix]
        img = show(split, [iid], 224)[0]
        axes[i, 0].imshow(img)
        axes[i, 0].set_ylabel(name, fontsize=6)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        for j, (k, t) in enumerate(cols, 1):
            rr = res[(k, t, 0)]
            jx = int(np.flatnonzero(rr["image_id"] == iid)[0])
            overlay(axes[i, j], img, rr["cam"][jx], f"{heads[j - 1]}\n" if i == 0 else "")
    for ax in axes[0]:
        ax.title.set_fontsize(6)
    fig.subplots_adjust(left=0.03, right=0.995, top=0.88, bottom=0.01, wspace=0.04, hspace=0.06)
    for ext in ("png", "pdf"):
        fig.savefig(EXPLAIN_DIR / f"paper_gradcam.{ext}", dpi=300)
    plt.close(fig)


def write_tex(df):
    """Rows for paper.tex: deletion ratio at FP32 and Spearman vs. FP32 at 8/4/2/1 bits, mean over seeds."""
    rows = []
    for key in [k for k in ORDER if k in set(df.model) and k not in E6_VARIANTS]:  # E6 dropped from the paper
        d = df[df.model == key].groupby("bits")
        cells = [f"{d.deletion_ratio.mean()['fp32']:.2f}" if "fp32" in d.groups else "--"]
        for t in BIT_ORDER[1:]:
            v = d.spearman_vs_fp32.mean().get(t, np.nan) if t in d.groups else np.nan
            cells.append("--" if np.isnan(v) else f"{v:.2f}")
        rows.append(f"{NAMES[key]} & " + " & ".join(cells) + " \\\\")
    (EXPLAIN_DIR / "table_explain.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ORDER, choices=ORDER)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--bits", type=int, nargs="+", default=[8, 4, 2, 1])
    ap.add_argument("--split", default="valB", choices=["valA", "valB", "test"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--report-only", action="store_true", help="skip computing maps; redo tables and figures")
    a = ap.parse_args()
    if not a.report_only:
        compute(a, device())
    res = load_all(a.split)
    if not res:
        sys.exit("no maps yet")
    df = summarize(res)
    df.to_csv(EXPLAIN_DIR / "summary.csv", index=False, float_format="%.4f")
    write_table(df, EXPLAIN_DIR / "table_explain.md")
    fig_stability(df)
    panels(res, a.split)
    paper_figure(res, a.split)
    write_tex(df)
    print((EXPLAIN_DIR / "table_explain.md").read_text())

"""E5 + all tables and figures, computed only from prediction files.

    python scripts/analyze.py            # writes analysis/*.csv, *.md, fig1.png, fig2.png
For each run: temperature on Val-A, conformal tau on Val-B (alpha 0.10 and 0.05), metrics on Test.
Compressed variants are scored twice: inherited (parent's T and tau) and recalibrated (their own).
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from tzq.config import ANALYSIS_DIR, MODELS, PROBES, QUANT_DIR, RUNS_DIR, TZ3  # noqa: E402
from tzq.metrics import (conformal_tau, ece, fit_temperature, macro_f1, referral_rates,  # noqa: E402
                         sensitivity, spec_threshold, stratified_bootstrap_indices, t1_auroc,
                         tz3_score)
from tzq.utils import load_json, logits_of, read_preds  # noqa: E402

ALPHAS = (0.10, 0.05)
NI_MARGIN = -0.03  # fixed before opening Test
NAMES = {
    "hsv": ("C1", "HSV + LR"), "dinov2_b@E1": ("P1", "DINOv2-B probe"), "dinov3_b@E1": ("P2", "DINOv3-B probe"),
    "medsiglip": ("P3", "MedSigLIP probe"), "resnet50": ("R1", "ResNet-50"), "deit_s": ("R2", "DeiT-S"),
    "dinov3_b": ("R3", "DINOv3-B fine-tuned"), "mnv4_m": ("R4", "MobileNetV4-Conv-M"),
    "tinyvit_11m": ("R5", "TinyViT-11M"), "binatt_deit_s": ("B1", "BinaryAttention (DeiT-S)"),
    "bhvit_tiny_fp": ("B3", "BHViT-Tiny"), "bhvit_tiny_bin": ("B2", "BHViT-Tiny"),
}
RUN_RE = re.compile(r"^(E\d[ab]?)_(.+)_(fp32|w\d|int8|bin|qk1)_s(\d+)$")


def discover():
    rows = []
    for d in sorted(list(RUNS_DIR.glob("E*_s*")) + list(QUANT_DIR.glob("E*_s*"))):
        m = RUN_RE.match(d.name)
        if not m or not all((d / f"preds_{s}.csv").exists() for s in ("valA", "valB", "test")):
            continue
        exp, model, bits, seed = m.groups()
        if exp == "E2" or exp == "E3a" or exp == "E3b":
            parent = None if exp == "E2" else f"E2_{model}_fp32_s{seed}"
        elif exp == "E4":
            # Full-precision parent whose T and tau the binary model inherits (set by e4_binary.py).
            parent = yaml.safe_load((d / "config.yaml").read_text()).get("parent")
        else:
            parent = None
        rows.append(dict(run=d.name, dir=d, exp=exp, model=model, bits=bits, seed=int(seed), parent=parent))
    return pd.DataFrame(rows)


def calibrate(d: Path):
    """Own temperature (Val-A) and tau (Val-B), plus everything needed later."""
    P = {s: read_preds(d, s) for s in ("valA", "valB", "test")}
    Z = {s: logits_of(p) for s, p in P.items()}
    Y = {s: p.label.to_numpy() for s, p in P.items()}
    T = fit_temperature(Z["valA"], Y["valA"])
    S = {s: tz3_score(Z[s], T) for s in Z}
    tau = {a: conformal_tau(Y["valB"], S["valB"], a) for a in ALPHAS}
    cut = spec_threshold(Y["valA"], S["valA"])
    return dict(Z=Z, Y=Y, T=T, S=S, tau=tau, cut=cut, ids=P["test"].image_id.to_numpy())


def test_metrics(c, idx=None, inherit=None):
    """Test metrics for one run on resample `idx`. `inherit` = parent's calibration dict."""
    y, z = c["Y"]["test"], c["Z"]["test"]
    if idx is not None:
        y, z = y[idx], z[idx]
    s = tz3_score(z, c["T"])
    out = dict(auroc=t1_auroc(y, s), sens80=sensitivity(y, s, c["cut"]), ece_raw=ece(y, tz3_score(z, 1.0)),
               ece=ece(y, s), f1=macro_f1(y, z))
    for a in ALPHAS:
        r = referral_rates(y, s, c["tau"][a])
        out[f"missed_{a}_recal"], out[f"elig_{a}_recal"] = r["missed_tz3"], r["eligibility"]
        if inherit is not None:
            r = referral_rates(y, tz3_score(z, inherit["T"]), inherit["tau"][a])
            out[f"missed_{a}_inh"], out[f"elig_{a}_inh"] = r["missed_tz3"], r["eligibility"]
    return out


def size_and_latency(row):
    d = row.dir
    info = dict(size_mb=np.nan, cpu_ms=np.nan)
    if (d / "cost.json").exists():
        c = load_json(d / "cost.json")
        info["size_mb"] = c.get("size_mb_int8", c.get("size_mb_simulated", c.get("size_mb_packed", np.nan)))
        info["gops_est"] = c.get("ops_g_estimated", np.nan)
        info["cpu_ms"] = c.get("latency_ms_int8", np.nan)
    elif (d / "model.pt").exists():
        info["size_mb"] = (d / "model.pt").stat().st_size / 1e6
    if row.exp == "E2":
        e3b = QUANT_DIR / f"E3b_{row.model}_int8_s0" / "cost.json"
        if e3b.exists():
            info["cpu_ms"] = load_json(e3b).get("latency_ms_fp32", np.nan)
    return info


def group_key(r):
    return (r.exp, r.model, r.bits)


def label(exp, model, bits):
    key = f"{model}@E1" if exp == "E1" and f"{model}@E1" in NAMES else model
    mid, name = NAMES.get(key, ("?", model))
    b = {"fp32": "32/32", "int8": "8/8", "bin": "1/1", "qk1": "32/32, QK 1-bit"}.get(bits, bits.replace("w", "") + "/32")
    return mid, name, b


def main(n_boot):
    runs = discover()
    if runs.empty:
        print("no complete runs found")
        return
    ANALYSIS_DIR.mkdir(exist_ok=True)
    cal = {r.run: calibrate(r.dir) for r in runs.itertuples()}
    ref_ids = next(iter(cal.values()))["ids"]
    for k, c in cal.items():
        assert (c["ids"] == ref_ids).all(), f"{k}: test rows are not aligned"
    y_test = next(iter(cal.values()))["Y"]["test"]
    boots = stratified_bootstrap_indices(y_test, n_boot, seed=0)

    # Per-run point estimates (E5 table).
    per_run = []
    for r in runs.itertuples():
        inh = cal.get(r.parent) if r.parent else None
        m = test_metrics(cal[r.run], inherit=inh)
        per_run.append(dict(run=r.run, exp=r.exp, model=r.model, bits=r.bits, seed=r.seed, parent=r.parent,
                            T=cal[r.run]["T"], **{f"tau_{a}": cal[r.run]["tau"][a] for a in ALPHAS}, **m,
                            **size_and_latency(r)))
    per_run = pd.DataFrame(per_run)
    per_run.to_csv(ANALYSIS_DIR / "e5_per_run.csv", index=False)

    # Bootstrap per group (seeds averaged inside each resample), plus paired difference to the parent group.
    groups = defaultdict(list)
    for r in runs.itertuples():
        groups[group_key(r)].append(r)
    boot_vals = {}
    for g, rs in groups.items():
        vals = []
        for idx in boots:
            ms = [test_metrics(cal[r.run], idx, cal.get(r.parent) if r.parent else None) for r in rs]
            vals.append({k: np.mean([m[k] for m in ms]) for k in ms[0]})
        boot_vals[g] = pd.DataFrame(vals)

    summary = []
    for g, rs in groups.items():
        pr = per_run[per_run.run.isin([r.run for r in rs])]
        row = dict(exp=g[0], model=g[1], bits=g[2], n_seeds=len(rs))
        mid, name, b = label(*g)
        row.update(id=mid, name=name, bits_label=b)
        for k in boot_vals[g].columns:
            row[k] = pr[k].mean()
            row[f"{k}_sd"] = pr[k].std(ddof=1) if len(pr) > 1 else np.nan
            row[f"{k}_lo"], row[f"{k}_hi"] = np.percentile(boot_vals[g][k], [2.5, 97.5])
        row["size_mb"], row["cpu_ms"] = pr.size_mb.mean(), pr.cpu_ms.mean()
        parent = rs[0].parent
        if parent:
            pg = next((k for k, v in groups.items() if parent in [x.run for x in v]), None)
            if pg:
                diff = boot_vals[g]["auroc"] - boot_vals[pg]["auroc"]
                row["d_auroc"] = row["auroc"] - per_run[per_run.run.isin([x.run for x in groups[pg]])].auroc.mean()
                row["d_auroc_lo"], row["d_auroc_hi"] = np.percentile(diff, [2.5, 97.5])
                row["non_inferior"] = bool(row["d_auroc_lo"] > NI_MARGIN)
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(ANALYSIS_DIR / "summary.csv", index=False)

    write_table1(summary)
    fig1(summary)
    fig2(summary)
    table2(summary, per_run, y_test)
    print("wrote", ANALYSIS_DIR)


def fmt(v, nd=3):
    return "" if pd.isna(v) else f"{v:.{nd}f}"


TABLE1_ORDER = [("E1", "hsv", "fp32"), ("E1", "dinov2_b", "fp32"), ("E1", "dinov3_b", "fp32"),
                ("E1", "medsiglip", "fp32"), ("E2", "resnet50", "fp32"), ("E2", "deit_s", "fp32"),
                ("E3b", "deit_s", "int8"), ("E2", "dinov3_b", "fp32"), ("E3b", "dinov3_b", "int8"),
                ("E2", "mnv4_m", "fp32"), ("E3b", "mnv4_m", "int8"), ("E2", "tinyvit_11m", "fp32"),
                ("E3b", "tinyvit_11m", "int8"), ("E4", "binatt_deit_s", "qk1"), ("E4", "bhvit_tiny_fp", "fp32"),
                ("E4", "bhvit_tiny_bin", "bin")]


def write_table1(summary):
    lines = ["| ID | Model | Bits (W/A) | Size (MB) | T1 AUROC [95% CI] | TZ3 sens. @ spec 0.80 | T1 ECE | "
             "T2 macro-F1 | CPU ms |", "|" + " --- |" * 9]
    for g in TABLE1_ORDER:
        r = summary[(summary.exp == g[0]) & (summary.model == g[1]) & (summary.bits == g[2])]
        if r.empty:
            continue
        r = r.iloc[0]
        sd = f" ±{r.auroc_sd:.3f}" if not pd.isna(r.auroc_sd) else ""
        lines.append(f"| {r.id} | {r['name']} | {r.bits_label} | {fmt(r.size_mb, 1)} | "
                     f"{r.auroc:.3f} [{r.auroc_lo:.3f}, {r.auroc_hi:.3f}]{sd} | {fmt(r.sens80)} | "
                     f"{fmt(r.ece)} (raw {fmt(r.ece_raw)}) | {fmt(r.f1)} | {fmt(r.cpu_ms, 1)} |")
    (ANALYSIS_DIR / "table1.md").write_text("\n".join(lines) + "\n")


def bits_series(summary, model):
    pts = []
    for bits, x in [("fp32", 32), ("w8", 8), ("w4", 4), ("w2", 2), ("w1", 1)]:
        exp = "E2" if bits == "fp32" else "E3a"
        r = summary[(summary.exp == exp) & (summary.model == model) & (summary.bits == bits)]
        if not r.empty:
            pts.append((x, r.iloc[0]))
    return pts


def fig1(summary):
    fig, ax = plt.subplots(figsize=(3.5, 2.6), dpi=200)
    for model in ["deit_s", "dinov3_b", "mnv4_m", "tinyvit_11m"]:
        pts = bits_series(summary, model)
        if not pts:
            continue
        x = [p[0] for p in pts]
        y = [p[1].auroc for p in pts]
        err = [[p[1].auroc - p[1].auroc_lo for p in pts], [p[1].auroc_hi - p[1].auroc for p in pts]]
        line = ax.errorbar(x, y, yerr=err, marker="o", ms=3, lw=1, capsize=2, label=NAMES[model][1])
        r = summary[(summary.exp == "E3b") & (summary.model == model)]
        if not r.empty:
            ax.plot([8], [r.iloc[0].auroc], marker="s", mfc="none", color=line[0].get_color(), ms=5)
    for model, mk in [("binatt_deit_s", "^"), ("bhvit_tiny_bin", "*")]:
        r = summary[(summary.exp == "E4") & (summary.model == model)]
        if not r.empty:
            ax.plot([1], [r.iloc[0].auroc], marker=mk, color="k", ms=6, ls="none", label=NAMES[model][1])
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 32], ["1", "2", "4", "8", "32"])
    ax.set_xlabel("Weight bit width")
    ax.set_ylabel("T1 AUROC (Test)")
    ax.legend(fontsize=5, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "fig1_auroc_vs_bits.png")
    fig.savefig(ANALYSIS_DIR / "fig1_auroc_vs_bits.pdf")


def fig2(summary, alpha=0.10):
    fig, ax = plt.subplots(figsize=(3.5, 2.6), dpi=200)
    for model in ["deit_s", "dinov3_b", "mnv4_m", "tinyvit_11m"]:
        pts = bits_series(summary, model)
        if not pts:
            continue
        x = [p[0] for p in pts]
        inh = [p[1].get(f"missed_{alpha}_inh", p[1][f"missed_{alpha}_recal"]) for p in pts]
        rec = [p[1][f"missed_{alpha}_recal"] for p in pts]
        ln = ax.plot(x, inh, marker="o", ms=3, lw=1, label=f"{NAMES[model][1]} inherited")
        ax.plot(x, rec, marker="o", ms=3, lw=1, ls="--", color=ln[0].get_color(), mfc="none")
    ax.axhline(alpha, color="k", lw=0.8, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 32], ["1", "2", "4", "8", "32"])
    ax.set_xlabel("Weight bit width (solid: inherited, dashed: recalibrated)")
    ax.set_ylabel("Missed-TZ3 rate (Test)")
    ax.legend(fontsize=5, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "fig2_missed_vs_bits.png")
    fig.savefig(ANALYSIS_DIR / "fig2_missed_vs_bits.pdf")


def table2(summary, per_run, y_test, alpha=0.10):
    """Outcomes per 1,000 examinations at Test prevalence; 'best' chosen on Val-A AUROC, never Test."""
    prev = (y_test == TZ3).mean()
    valA_auc = {r.run: load_valA_auroc(r) for r in per_run.itertuples()}
    per_run = per_run.assign(valA_auroc=per_run.run.map(valA_auc))
    families = {"Full precision": per_run.bits == "fp32",
                "INT8": per_run.bits == "int8",
                "Binary": per_run.bits.isin(["bin", "qk1"])}
    lines = ["| Model | Regime | Eligible | of which TZ3 | Referred | of which TZ1/2 |", "|" + " --- |" * 6]
    for fam, mask in families.items():
        sub = per_run[mask]
        if sub.empty:
            continue
        g = sub.groupby(["exp", "model", "bits"]).valA_auroc.mean().idxmax()
        r = summary[(summary.exp == g[0]) & (summary.model == g[1]) & (summary.bits == g[2])].iloc[0]
        for regime in ["recal", "inh"]:
            if f"missed_{alpha}_{regime}" not in r or pd.isna(r[f"missed_{alpha}_{regime}"]):
                continue
            miss, elig = r[f"missed_{alpha}_{regime}"], r[f"elig_{alpha}_{regime}"]
            e_pos, e_neg = 1000 * prev * miss, 1000 * (1 - prev) * elig
            lines.append(f"| {fam}: {r['name']} {r.bits_label} | {'recalibrated' if regime == 'recal' else 'inherited'} | "
                         f"{e_pos + e_neg:.0f} | {e_pos:.0f} | {1000 - e_pos - e_neg:.0f} | {1000 * (1 - prev) - e_neg:.0f} |")
    (ANALYSIS_DIR / "table2.md").write_text(f"alpha = {alpha}, Test prevalence of TZ3 = {prev:.1%}\n\n"
                                            + "\n".join(lines) + "\n")


def load_valA_auroc(r):
    d = (RUNS_DIR if r.exp in ("E1", "E2", "E4") else QUANT_DIR) / r.run
    p = read_preds(d, "valA")
    return t1_auroc(p.label, tz3_score(logits_of(p)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    main(ap.parse_args().n_boot)

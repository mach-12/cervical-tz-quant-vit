"""Check that every run and result file the paper needs is on Drive, and say which script to run for what is missing.

    python scripts/check_outputs.py
Reads files only (no GPU, no models loaded). OK = complete; MISSING = not there; PARTIAL = started but unfinished;
FAILED = the script wrote FAILED.txt; STALE = computed from an older checkpoint than the parent's current model.pt.
Optional runs (448 px ordinal) are reported but do not count as missing.
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tzq.config import ANALYSIS_DIR, EXPLAIN_DIR, PARENTS, QUANT_DIR, ROOT, RUNS_DIR  # noqa: E402
from tzq.utils import finished, stale  # noqa: E402

SEEDS = [0, 1, 2]
PREDS = [f"preds_{s}.csv" for s in ("valA", "valB", "test")]
E6 = ["dinov3_b_cont", "dinov3_b_ord"]
OPTIONAL = ["dinov3_b_ord448"]
problems = []


def has_preds(d):
    return all((d / f).exists() for f in PREDS)


def report(section, name, status, fix=None, optional=False):
    if status != "OK" and not optional:
        problems.append((section, name, status, fix))
    tag = status if not optional or status == "OK" else f"{status} (optional)"
    print(f"  {tag:<20} {name}" + (f"   -> {fix}" if status != "OK" and fix else ""))


def check_train(d, fix, optional=False):
    if finished(d):
        status = "OK"
    elif d.exists():
        status = "PARTIAL"
    else:
        status = "MISSING"
    report("train", d.name, status, fix, optional)


def check_quant(d, parent, fix, optional=False):
    if (d / "FAILED.txt").exists() and not has_preds(d):
        status = "FAILED"
    elif not has_preds(d):
        status = "PARTIAL" if d.exists() else "MISSING"
    elif stale(d / "preds_test.csv", parent):
        status = "STALE"
    else:
        status = "OK"
    report("quant", d.name, status, fix, optional)


def check_file(f, newer_than=None):
    if not f.exists():
        report("files", str(f.relative_to(ROOT)), "MISSING")
        return
    t = datetime.fromtimestamp(f.stat().st_mtime)
    if newer_than and f.stat().st_mtime < newer_than:
        report("files", f"{f.relative_to(ROOT)} ({t:%b %d %H:%M})", "STALE",
               "older than the newest run: rerun analyze.py / explain.py")
    else:
        report("files", f"{f.relative_to(ROOT)} ({t:%b %d %H:%M})", "OK")


if __name__ == "__main__":
    print("project folder:", ROOT)
    if not RUNS_DIR.exists():
        sys.exit(f"no runs/ folder in {ROOT}: wrong folder? (cd to the Drive repo or set TZQ_ROOT)")

    print("\nE1 probes")
    for p in ("hsv", "dinov2_b", "dinov3_b", "medsiglip"):
        check_train(RUNS_DIR / f"E1_{p}_fp32_s0", f"e1_probe.py --probe {p}")

    print("\nE2 fine-tuning")
    check_train(RUNS_DIR / "E2_resnet50_fp32_s0", "e2_finetune.py --model resnet50")
    for m in PARENTS:
        for s in SEEDS:
            check_train(RUNS_DIR / f"E2_{m}_fp32_s{s}", f"e2_finetune.py --model {m}")

    print("\nE6 ordinal head")
    for m in E6 + OPTIONAL:
        for s in SEEDS:
            check_train(RUNS_DIR / f"E2_{m}_fp32_s{s}", f"e6_r3_ordinal.py --variants {m}", m in OPTIONAL)

    parents = PARENTS + E6
    print("\nE3a simulated weight quantization (8/4/2/1 bit)")
    for m in parents + OPTIONAL:
        for s in SEEDS:
            parent = RUNS_DIR / f"E2_{m}_fp32_s{s}"
            if not finished(parent):
                continue  # reported above
            for b in (8, 4, 2, 1):
                check_quant(QUANT_DIR / f"E3a_{m}_w{b}_s{s}", parent, f"e3a_weight_sweep.py --models {m}",
                            m in OPTIONAL)

    print("\nE3b ONNX INT8")
    for m in parents + OPTIONAL:
        for s in SEEDS:
            parent = RUNS_DIR / f"E2_{m}_fp32_s{s}"
            if not finished(parent):
                continue
            check_quant(QUANT_DIR / f"E3b_{m}_int8_s{s}", parent, f"e3b_onnx_int8.py --models {m}", m in OPTIONAL)

    print("\nE4 binarized attention")
    check_train(RUNS_DIR / "E4_binatt_deit_s_qk1_s0", "e4_binary.py binatt")

    runs = [p for d in (RUNS_DIR, QUANT_DIR) if d.exists() for p in d.glob("E*_s*/preds_test.csv")]
    newest = max((p.stat().st_mtime for p in runs), default=None)
    print(f"\nanalysis/ (newest run result: {datetime.fromtimestamp(newest):%b %d %H:%M})" if newest else "\nanalysis/")
    for f in ("summary.csv", "table1.tex", "table2.tex", "table_e6.tex", "fig1_auroc_vs_bits.pdf",
              "fig2_missed_vs_bits.pdf", "fig3_confusion.pdf"):
        check_file(ANALYSIS_DIR / f, newest)
    if (ANALYSIS_DIR / "table1.tex").exists() and "4/32" not in (ANALYSIS_DIR / "table1.tex").read_text():
        report("files", "analysis/table1.tex has no 4/2/1-bit rows", "STALE", "git pull, then rerun analyze.py")

    print("\nexplain/")
    for f in ("summary.csv", "table_explain.tex", "paper_gradcam.pdf"):
        check_file(EXPLAIN_DIR / f, newest)

    print("\n" + ("=" * 60))
    if not problems:
        print("ALL GOOD: everything the paper needs is here. Run collect_paper_files.py and download the zip.")
    else:
        print(f"{len(problems)} problem(s). Scripts to run (in this order):")
        order = ["e1", "e2", "e6", "e3a", "e3b", "e4", "analyze", "explain"]
        fixes = sorted({f for *_, f in problems if f}, key=lambda f: next(
            (i for i, k in enumerate(order) if f.startswith(k)), 99))
        for f in fixes:
            print("   python scripts/" + f if not f.startswith(("git", "older")) else "   " + f)
        print("then: python scripts/analyze.py, python scripts/explain.py, python scripts/collect_paper_files.py")

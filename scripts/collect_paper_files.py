"""Collect the result files paper.tex reads into one folder and a zip, for upload to Overleaf.

    python scripts/collect_paper_files.py            # -> paper_files/ and paper_files.zip (next to analysis/)
Upload the zip's contents into the Overleaf project next to paper.tex (keep the analysis/ and explain/ folders).
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tzq.config import ANALYSIS_DIR, EXPLAIN_DIR, ROOT  # noqa: E402

FILES = [ANALYSIS_DIR / f for f in ("table1.tex", "table2.tex", "fig1_auroc_vs_bits.pdf",
                                    "fig2_missed_vs_bits.pdf", "fig3_confusion.pdf", "table_e6.tex")] + \
        [EXPLAIN_DIR / f for f in ("table_explain.tex", "paper_gradcam.pdf")]

if __name__ == "__main__":
    out = ROOT / "paper_files"
    shutil.rmtree(out, ignore_errors=True)
    for f in FILES:
        if f.exists():
            dst = out / f.parent.name / f.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            print("copied ", dst.relative_to(out))
        else:
            print("missing", f.relative_to(ROOT), "(paper.tex shows a placeholder)")
    if out.exists():
        print("wrote", shutil.make_archive(str(out), "zip", out))

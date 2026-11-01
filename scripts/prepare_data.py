"""Download the dataset, build the square image cache, and (once) write the frozen splits.

    python scripts/prepare_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tzq.config import SPLITS_DIR  # noqa: E402
from tzq.data import build_cache, download, make_splits  # noqa: E402

if __name__ == "__main__":
    raw = download()
    print("cached images:", build_cache(raw))
    if (SPLITS_DIR / "valA.csv").exists():
        print("splits already frozen in", SPLITS_DIR)
    else:
        counts = make_splits(raw)
        for k, v in counts.items():
            print(f"{k:6s} n={sum(v):5d}  TZ1/TZ2/TZ3 = {v}")

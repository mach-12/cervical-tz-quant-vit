"""Download the dataset, build the square image cache, and (once) write the frozen splits.

    python scripts/prepare_data.py
If the 512 px cache already holds every image in the frozen splits (e.g. restored from Google Drive),
nothing is downloaded.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tzq.config import SPLITS, SPLITS_DIR  # noqa: E402
from tzq.data import CACHE_DIR, build_cache, download, load_split, make_splits  # noqa: E402


def cache_complete():
    if not (SPLITS_DIR / "valA.csv").exists():
        return False
    return all((CACHE_DIR / p).exists() for s in SPLITS for p in load_split(s).path)


if __name__ == "__main__":
    if cache_complete():
        print("image cache complete:", CACHE_DIR)
        sys.exit()
    raw = download()
    print("cached images:", build_cache(raw))
    if (SPLITS_DIR / "valA.csv").exists():
        print("splits already frozen in", SPLITS_DIR)
    else:
        counts = make_splits(raw)
        for k, v in counts.items():
            print(f"{k:6s} n={sum(v):5d}  TZ1/TZ2/TZ3 = {v}")

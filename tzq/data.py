"""Dataset download, square cache, frozen splits, and transforms."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .config import CACHE_SIZE, CLASSES, DATA_DIR, HF_DATASET, SPLITS_DIR

RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / f"sq{CACHE_SIZE}"


def download(raw_dir: Path = RAW_DIR) -> Path:
    from huggingface_hub import snapshot_download
    snapshot_download(HF_DATASET, repo_type="dataset", local_dir=raw_dir, max_workers=16)
    return raw_dir


def pad_square(img: Image.Image) -> Image.Image:
    """Pad to square with black, keeping every pixel of the crop unchanged."""
    w, h = img.size
    s = max(w, h)
    if w == h:
        return img
    out = Image.new("RGB", (s, s), (0, 0, 0))
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


def _cache_one(args):
    src, dst = args
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = pad_square(Image.open(src).convert("RGB"))
    img.resize((CACHE_SIZE, CACHE_SIZE), Image.BICUBIC).save(dst, quality=95)


def build_cache(raw_dir: Path = RAW_DIR, cache_dir: Path = CACHE_DIR, workers: int = 8):
    """Square-pad every crop and store it at CACHE_SIZE px, so training does not decode full-size JPEGs."""
    jobs = [(p, cache_dir / p.relative_to(raw_dir)) for p in sorted(raw_dir.glob("*/Type_*/*.jpg"))]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(_cache_one, jobs))
    return len(jobs)


def list_images(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    rows = []
    for p in sorted(raw_dir.glob("*/Type_*/*.jpg")):
        split, cls = p.parts[-3], p.parts[-2]
        stem = p.stem
        rows.append(dict(
            image_id=f"{split}_{cls}_{stem}",
            path=str(p.relative_to(raw_dir)),
            label=CLASSES.index(cls),
            source_split=split,
            group=f"{split}_{cls}_{stem.split('__crop')[0]}",  # crops of one photo share a group
        ))
    return pd.DataFrame(rows)


def make_splits(raw_dir: Path = RAW_DIR, out_dir: Path = SPLITS_DIR, seed: int = 0):
    """Write splits/{train,valA,valB,test}.csv.

    The published split has no cluster IDs, so validation is divided 50/50 stratified by
    label (protocol option 2), keeping crops of the same photo in the same half.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    out_dir.mkdir(parents=True, exist_ok=True)
    if any((out_dir / f"{s}.csv").exists() for s in ["train", "valA", "valB", "test"]):
        raise FileExistsError(f"{out_dir} already holds split files; they are frozen and never regenerated.")
    df = list_images(raw_dir)
    df["cluster_id"] = ""
    val = df[df.source_split == "val"].reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    a_idx, b_idx = next(sgkf.split(val, val.label, val.group))
    parts = {
        "train": df[df.source_split == "train"],
        "valA": val.iloc[b_idx],
        "valB": val.iloc[a_idx],
        "test": df[df.source_split == "test"],
    }
    cols = ["image_id", "path", "label", "cluster_id", "group"]
    for name, part in parts.items():
        part.sort_values("image_id")[cols].to_csv(out_dir / f"{name}.csv", index=False)
    return {k: v.label.value_counts().sort_index().tolist() for k, v in parts.items()}


def load_split(split: str) -> pd.DataFrame:
    return pd.read_csv(SPLITS_DIR / f"{split}.csv", keep_default_na=False)


# ---------------------------------------------------------------- transforms

def eval_transform(size, mean, std):
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def train_transform(size, mean, std):
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1),
                            interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(0.5),
        T.RandomRotation(15, interpolation=T.InterpolationMode.BILINEAR),
        T.ColorJitter(0.2, 0.2, 0.1, 0.02),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


class CervixDataset(Dataset):
    def __init__(self, split: str, transform, image_dir: Path = CACHE_DIR, limit: int | None = None):
        self.df = load_split(split)
        if limit:
            self.df = self.df.groupby("label", group_keys=False).head(limit // 3).reset_index(drop=True)
        self.split = split
        self.transform = transform
        self.image_dir = Path(image_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.image_dir / r.path).convert("RGB")
        return self.transform(img), int(r.label), i


def make_loader(split, transform, batch_size=64, shuffle=False, workers=4, limit=None, drop_last=False):
    ds = CervixDataset(split, transform, limit=limit)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=drop_last,
                      persistent_workers=workers > 0)

"""Devices, seeding, prediction files, and the Test-access log."""
import datetime as dt
import json
import random
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ANALYSIS_DIR
from .data import load_split


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast(dev):
    """bf16 where supported (A100/L4/H100), fp16 on older GPUs such as the T4, off elsewhere."""
    if dev.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype), dtype
    return torch.autocast(dev.type, enabled=False), torch.float32


@torch.no_grad()
def predict_logits(model, loader, dev):
    model.eval()
    ctx, _ = autocast(dev)
    out = np.zeros((len(loader.dataset), 3), dtype=np.float32)
    for x, _, idx in loader:
        with ctx:
            z = model(x.to(dev, non_blocking=True))
        out[idx.numpy()] = z.float().cpu().numpy()
    return out


def write_preds(run_dir: Path, split: str, logits: np.ndarray):
    df = load_split(split)
    assert len(df) == len(logits), (split, len(df), len(logits))
    out = pd.DataFrame({
        "image_id": df.image_id, "split": split, "label": df.label,
        "logit_tz1": logits[:, 0], "logit_tz2": logits[:, 1], "logit_tz3": logits[:, 2],
    })
    run_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(run_dir / f"preds_{split}.csv", index=False, float_format="%.6f")
    if split == "test":
        log_test_access(run_dir.name)


def read_preds(run_dir: Path, split: str) -> pd.DataFrame:
    return pd.read_csv(Path(run_dir) / f"preds_{split}.csv")


def logits_of(df: pd.DataFrame) -> np.ndarray:
    return df[["logit_tz1", "logit_tz2", "logit_tz3"]].to_numpy(np.float64)


def log_test_access(run_id: str):
    """Protocol rule: Test is opened once per final model, and each opening is logged."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    path = ANALYSIS_DIR / "test_access_log.csv"
    new = not path.exists()
    with open(path, "a") as f:
        if new:
            f.write("run_id,utc_time,host\n")
        f.write(f"{run_id},{dt.datetime.now(dt.timezone.utc).isoformat()},{socket.gethostname()}\n")


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def finished(run_dir: Path) -> bool:
    """A training run is finished once its summary and Test predictions exist. model.pt alone is not enough:
    it is rewritten during training whenever Val-A improves, so it also exists for interrupted runs."""
    return (Path(run_dir) / "summary.json").exists() and (Path(run_dir) / "preds_test.csv").exists()


def stale(result: Path, parent: Path) -> bool:
    """True if `result` (a file) was computed from an older checkpoint than parent/model.pt."""
    result, ckpt = Path(result), Path(parent) / "model.pt"
    return result.exists() and ckpt.exists() and result.stat().st_mtime < ckpt.stat().st_mtime

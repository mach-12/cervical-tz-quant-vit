"""E3b: real INT8 (QDQ, per-channel, static MinMax calibration on 256 Train images) with ONNX Runtime.

    python scripts/e3b_onnx_int8.py --models deit_s mnv4_m tinyvit_11m dinov3_b
Latency (batch 1, 4 threads, 50 warm-up, median of 200) is measured for seed 0, FP32 and INT8,
on whatever machine runs this script; its CPU is recorded.
"""
import argparse
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,  # noqa: E402
                                      QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process  # noqa: E402

from tzq.config import PARENTS, QUANT_DIR, RUNS_DIR  # noqa: E402
from tzq.data import CervixDataset, eval_transform, make_loader  # noqa: E402
from tzq.metrics import t1_auroc, tz3_score  # noqa: E402
from tzq.models import build_classifier  # noqa: E402
from tzq.utils import read_preds, logits_of, save_json, write_preds  # noqa: E402

N_CALIB, MAX_DROP = 256, 0.02


def cpu_name():
    try:
        if platform.system() == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


def export(model, size, path: Path):
    model.eval()
    x = torch.randn(1, 3, size, size)
    try:
        torch.onnx.export(model, (x,), str(path), opset_version=17, input_names=["x"], output_names=["logits"],
                          dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
    except Exception:
        batch = torch.export.Dim("batch", min=1, max=256)
        torch.onnx.export(model, (x,), str(path), opset_version=17, input_names=["x"], output_names=["logits"],
                          dynamic_shapes={"x": {0: batch}}, dynamo=True, external_data=False)


class Reader(CalibrationDataReader):
    def __init__(self, tf):
        ds = CervixDataset("train", tf)
        rng = np.random.default_rng(0)
        self.idx = iter(rng.choice(len(ds), N_CALIB, replace=False).tolist())
        self.ds = ds

    def get_next(self):
        i = next(self.idx, None)
        return None if i is None else {"x": self.ds[i][0].unsqueeze(0).numpy()}


def session(path, threads=None):
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def run_onnx(sess, split, tf):
    dl = make_loader(split, tf, 32, workers=2)
    out = np.zeros((len(dl.dataset), 3), np.float32)
    for x, _, idx in dl:
        out[idx.numpy()] = sess.run(None, {"x": x.numpy()})[0]
    return out, dl.dataset.df.label.to_numpy()


def latency_ms(path, size, threads=4, warmup=50, n=200):
    sess = session(path, threads)
    x = np.random.default_rng(0).standard_normal((1, 3, size, size)).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {"x": x})
    times = []
    for _ in range(n):
        t = time.perf_counter()
        sess.run(None, {"x": x})
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def one(key, seed, measure_latency):
    parent = RUNS_DIR / f"E2_{key}_fp32_s{seed}"
    out = QUANT_DIR / f"E3b_{key}_int8_s{seed}"
    if (out / "preds_test.csv").exists() or not (parent / "model.pt").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    model, size, mean, std = build_classifier(key, pretrained=False)
    model.load_state_dict(torch.load(parent / "model.pt", map_location="cpu"))
    tf = eval_transform(size, mean, std)
    fp32, pre = out / "fp32.onnx", out / "fp32_pre.onnx"
    export(model, size, fp32)
    try:
        quant_pre_process(str(fp32), str(pre))
    except Exception:
        pre = fp32
    pa = read_preds(parent, "valA")
    parent_auc = t1_auroc(pa.label, tz3_score(logits_of(pa)))

    tried = {}
    for method in [CalibrationMethod.MinMax, CalibrationMethod.Percentile]:
        q = out / f"int8_{method.name.lower()}.onnx"
        quantize_static(str(pre), str(q), Reader(tf), quant_format=QuantFormat.QDQ, per_channel=True,
                        weight_type=QuantType.QInt8, activation_type=QuantType.QInt8, calibrate_method=method)
        z, y = run_onnx(session(q), "valA", tf)
        tried[method.name] = dict(path=q, auc=t1_auroc(y, tz3_score(z)), z=z)
        print(f"{out.name} {method.name}: valA AUROC {tried[method.name]['auc']:.4f} (parent {parent_auc:.4f})")
        if parent_auc - tried[method.name]["auc"] <= MAX_DROP:
            break
    best = max(tried.values(), key=lambda r: r["auc"])
    int8 = out / "int8.onnx"
    best["path"].replace(int8)

    sess = session(int8)
    write_preds(out, "valA", best["z"])
    for split in ["valB", "test"]:
        write_preds(out, split, run_onnx(sess, split, tf)[0])
    info = dict(parent=parent.name, parent_valA_auroc=parent_auc,
                calibration={k: v["auc"] for k, v in tried.items()},
                chosen=[k for k, v in tried.items() if v is best][0],
                size_mb_fp32=fp32.stat().st_size / 1e6, size_mb_int8=int8.stat().st_size / 1e6)
    if measure_latency:
        info.update(cpu=cpu_name(), threads=4, latency_ms_fp32=latency_ms(fp32, size),
                    latency_ms_int8=latency_ms(int8, size), onnxruntime=ort.__version__)
    save_json(out / "cost.json", info)
    for f in out.glob("*.onnx"):
        if f.name not in ("fp32.onnx", "int8.onnx"):
            f.unlink()
    print(info)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["deit_s", "mnv4_m", "tinyvit_11m", "dinov3_b"],
                    choices=PARENTS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    for key in a.models:
        for seed in a.seeds:
            try:
                one(key, seed, measure_latency=seed == 0)
            except Exception:
                # Protocol: try R3 and record it if export fails.
                (QUANT_DIR / f"E3b_{key}_int8_s{seed}").mkdir(parents=True, exist_ok=True)
                (QUANT_DIR / f"E3b_{key}_int8_s{seed}" / "FAILED.txt").write_text(traceback.format_exc())
                print(f"E3b {key} s{seed} failed; see FAILED.txt")

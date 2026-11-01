# Quantized and Binary Vision Transformers for Cervical TZ Classification

Code for the protocol in `Experiment Protocol Quantized and Binary Vision Transformers for Cervical Transf.md`
(its "Implementation notes" section lists every place the code departs from or pins down the protocol).

## Layout

| Path | Contents |
| --- | --- |
| `tzq/` | Library: config, data and splits, models, training loop, quantization, metrics, cost |
| `scripts/prepare_data.py` | Download `mach-12/cervical_cancer_classification`, build the 512 px square cache, write the frozen splits (once) |
| `scripts/e1_probe.py` | E1: HSV, DINOv2-B, DINOv3-B, MedSigLIP logistic-regression probes |
| `scripts/e2_finetune.py` | E2: fine-tuning R1–R5 (LR search on seed 0, then seeds 1–2, then Test once) |
| `scripts/e3a_weight_sweep.py` | E3a: simulated 8/4/2/1-bit weight quantization |
| `scripts/e3b_onnx_int8.py` | E3b: ONNX Runtime INT8 (QDQ, per-channel, static), size and CPU latency |
| `scripts/e4_binary.py` | E4: BinaryAttention (B1), BHViT-Tiny 32/32 (B3) and 1/1 (B2) |
| `scripts/cost_fp.py` | GMACs of the full-precision models |
| `scripts/analyze.py` | E5 (temperature, conformal threshold, inherited vs. recalibrated) and every table and figure |
| `notebooks/colab_driver.ipynb` | Runs everything on a Colab GPU, keeping runs on Google Drive |
| `splits/` | Frozen `train/valA/valB/test.csv` — never regenerate |
| `runs/`, `quant/` | One folder per run: `config.yaml`, `preds_{valA,valB,test}.csv`, `model.pt`, `summary.json` / `cost.json` |
| `analysis/` | `summary.csv`, `e5_per_run.csv`, `table1.md`, `table2.md`, `fig1_*`, `fig2_*`, `test_access_log.csv` |

## Order

```bash
python scripts/prepare_data.py
python scripts/e1_probe.py                          # MedSigLIP needs its licence accepted on Hugging Face
python scripts/e2_finetune.py --model dinov3_b      # then deit_s, tinyvit_11m, mnv4_m, resnet50
python scripts/e3a_weight_sweep.py
python scripts/e3b_onnx_int8.py                     # CPU; latency is measured on this machine
python scripts/e4_binary.py binatt
python scripts/e4_binary.py bhvit_fp
python scripts/e4_binary.py bhvit_bin               # teacher = best R3 seed
python scripts/cost_fp.py
python scripts/analyze.py
```

Every script is resumable: finished runs (those with prediction files) are skipped.
Set `TZQ_ROOT` to keep splits/runs/analysis elsewhere (e.g. Google Drive) and `TZQ_DATA` for the image cache.

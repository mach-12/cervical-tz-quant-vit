# Experiment Protocol: Quantized and Binary Vision Transformers for Cervical Transformation Zone Classification

## Summary

This is the 6-page version: 12 models, their quantized variants and five experiments, answering one question about cervical transformation-zone (TZ) classification — how accuracy and calibration change as vision transformers go from 32-bit weights to 1-bit.

**Paper title:** *Quantized and Binary Vision Transformers for Cervical Transformation Zone Classification*

Alternatives in the same style:

- *Accuracy and Calibration of Low-Bit Vision Transformers for Cervical Transformation Zone Classification*
- *Low-Bit Vision Transformers for Cervical Transformation Zone Classification from Smartphone Images*

**Research question.** When a vision transformer that separates TZ3 (refer) from TZ1/2 (ablation-eligible) is quantized or binarized, how much accuracy is lost, and does a referral threshold set on the full-precision model still bound the share of missed TZ3 cases?

**Contributions**

1. MobileODT reframed correctly as TZ classification, with TZ3 sensitivity as the primary endpoint.
2. A benchmark from 32-bit foundation models to 1-bit transformers on one fixed split.
3. A calibration analysis showing whether compressed models must be recalibrated before their referral threshold can be trusted.

**Cut from the full protocol** (still documented in Experiment Protocol: Compression vs. Safety in Smartphone TZ Triage): extra probes (C-RADIOv3, CLIP, SigLIP, BiomedCLIP, DINOv2 S/L, DINOv3 S/L), the XGBoost baseline, Swin-T, ConvNeXt-T, fine-tuned DINOv2-S, LeViT, EfficientNet-B0, distillation, FIMA-Q, 1-bit embeddings, GSB, 4-bit quantization-aware training, zero-shot tests, NetScore, and reliability diagrams.

## Data

All experiments use 8,188 MobileODT cervix images in your published cluster split, with manual cervix crops; the only new step is dividing validation in two.

**Source.** Intel–MobileODT Cervical Cancer Screening (Kaggle, 2017): smartphone colposcope images from 21 countries, mostly East Africa and Mexico. Train, stage-1 test (labels from the released solution) and the additional set were merged: 8,686 images, 8,188 after removing images with no usable cervix.

**Labels (IFCPC transformation-zone types)**

- **TZ1:** fully on the ectocervix and fully visible.
- **TZ2:** has an endocervical component, but fully visible.
- **TZ3:** has an endocervical component that is not fully visible; not eligible for ablation, so referred.

**Split** (on Hugging Face as `mach-12/cervical_cancer_classification`; the code downloads this copy, whose train/val/test counts match the table exactly. The `mach-12/cervical_cancer_classification_clusters` copy has different counts — 8,020 crops, train/val/test = 6,422/754/844 — and is not used.)

| Split | Images | TZ1 | TZ2 | TZ3 | Used for |
| --- | --- | --- | --- | --- | --- |
| Train | 6,505 | 1,169 | 3,474 | 1,862 | Training, probe fitting, INT8 calibration |
| Val-A | \~415 | \~73 | \~219 | \~123 | Model selection, early stopping, temperature scaling |
| Val-B | \~416 | \~74 | \~219 | \~123 | Referral threshold only |
| Test | 852 | 199 | 378 | 275 | Final evaluation, once per final model |
| **Total** | **8,188** | **1,515** | **4,290** | **2,383** |  |

Validation came from 20 of 100 HSV-histogram K-means clusters (k = 100, seed 1312); Test is the official set, unchanged.

**Dividing validation**

1. ~~Assign the 20 validation clusters at random, 10 per half (seed 0); redraw until each half holds 110–136 TZ3 images.~~ Not possible: the published split carries no cluster IDs.
2. **Used.** Without cluster IDs, split the 831 images 50/50 stratified by label (seed 0) and say so. Implemented with `StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=0)`, grouping the 44 `crop1`/`crop2` pairs cut from the same photo so both crops land in the same half (`tzq/data.py: make_splits`).
3. Commit `splits/{train,valA,valB,test}.csv` (`image_id, path, label, cluster_id, group`; `cluster_id` is empty) and never regenerate them. `make_splits` refuses to overwrite existing files.

**Crops.** Bounding box of the Label Studio cervix mask, plus a 5% margin per side, padded to square; pixels inside the box are kept as they are. *Implementation:* the Hugging Face images are already the crops; `scripts/prepare_data.py` pads each to square with black and stores a 512 px bicubic copy (`data/sq512/`) so training does not decode full-size JPEGs. Every model input is resized from this copy (one extra resampling step). C1 histograms are computed on the unpadded original crop.

**Limitations to state:** possible near-duplicates between train and test, noisy labels in the additional set, no histology. Check the Kaggle rules on redistribution before keeping the cleaned copy public.

## Tasks and labels

Every model is trained once as a 3-class classifier, and both tasks are read from that output.

| Task | Definition | Score | Role |
| --- | --- | --- | --- |
| **T1** (primary) | TZ3 (positive) vs. TZ1/2 | s = p(TZ3) after temperature scaling | Ablation eligibility vs. referral; used for the referral threshold |
| **T2** (secondary) | TZ1 vs. TZ2 vs. TZ3 | Softmax over 3 classes | Comparability with prior MobileODT work |

- **Encoding:** `Type_1` → 0, `Type_2` → 1, `Type_3` → 2.
- **Class weights** (w\_c = N / (3 · n\_c) on Train): TZ1 = 1.855, TZ2 = 0.624, TZ3 = 1.165, for every trained model and probe.
- **Decision rule:** eligible for ablation if s < τ, otherwise refer; τ is set in E5.

## Setup

Three environments, one repository layout and one prediction-file format cover every experiment; all metrics are computed afterwards from the prediction files.

**Environments**

| Environment | Contents | For |
| --- | --- | --- |
| `env-main` | Python 3.11, PyTorch ≥ 2.5 (CUDA), torchvision, timm ≥ 1.0, transformers ≥ 4.56, scikit-learn, onnx, onnxruntime, fvcore, pandas, matplotlib | E1, E2, E3, E5 |
| `env-bhvit` | Python 3.11.7, PyTorch 2.5.1, CUDA 12.1, timm, transformers 4.39.2 (pinned by the BHViT repo) | BHViT |
| `env-binatt` | As in the BinaryAttention README | BinaryAttention |

Before starting, accept the Hugging Face licences for DINOv3 and MedSigLIP (both gated), then run `hf auth login`.

*Implementation:* one environment covers everything (Python 3.11, PyTorch ≥ 2.5, timm 1.0.30, transformers 5.x; `requirements.txt`). BHViT's model code imports only stable `transformers`/`timm` APIs, and BinaryAttention's timm 0.4.12 imports resolve through timm's compatibility shims, so `env-bhvit` and `env-binatt` are kept only as fallbacks if a model fails to build. DINOv3 is loaded from timm's ungated mirror of the same LVD-1689M weights (`vit_base_patch16_dinov3.lvd1689m`), so only MedSigLIP still needs a licence. Status on 2026-11-01: account `mach-12` gets 403 on `google/medsiglip-448` (licence not yet accepted).

**Repository**

- `splits/` — the four frozen CSVs.
- `features/<model>/<split>.npy` — frozen embeddings for E1.
- `runs/<run_id>/` — `config.yaml`, `preds_{valA,valB,test}.csv`, `model.pt`; run ID `<exp>_<model>_<bits>_s<seed>`, e.g. `E2_deit_s_fp32_s0`. LR-search runs live in `runs/lrsearch/<run_id>_lr<lr>/` and carry Val-A/Val-B predictions only; the chosen seed-0 run is copied to `runs/<run_id>/`.
- `quant/<run_id>/` — compressed variants, same files.
- `analysis/` — scripts that write every table and figure from the prediction files.

**Prediction file.** Store logits, because temperature and threshold are fitted later.

```csv
image_id,split,label,logit_tz1,logit_tz2,logit_tz3
Type_1_0001,test,0,2.41,0.37,-1.12
```

**Preprocessing**

- **Evaluation:** resize the square crop to the model's native size R (224 px; MedSigLIP 448 px; MobileNetV4 as in its timm config), bicubic, no centre crop, normalized with the checkpoint's own mean and std.
- **Training augmentation (E2, E4):** `RandomResizedCrop(R, scale=(0.7, 1.0), ratio=(0.9, 1.1))`, horizontal flip (p = 0.5), rotation ±15°, `ColorJitter(0.2, 0.2, 0.1, 0.02)`. No mixup or cutmix.

**Rules:** select on Val-A only; open Test once per final model and log the time; 3 seeds (0, 1, 2) for the four quantization parents, 1 seed elsewhere. *Implementation:* every Test prediction file written appends a line to `analysis/test_access_log.csv` (run ID, UTC time, host).

## Models

Twelve models, plus the quantized variants of four parents built in E3. Confirm each timm tag with `timm.list_pretrained('<name>*')` before running; if one is missing, use the closest listed tag and record it.

| ID | Model | Checkpoint | Training | Role |
| --- | --- | --- | --- | --- |
| C1 | HSV histogram + logistic regression | 32 bins per channel = 96 features | E1 | Lower bound |
| P1 | DINOv2 ViT-B/14 | `facebook/dinov2-base` | E1, frozen probe | General self-supervised features |
| P2 | DINOv3 ViT-B/16 | `facebook/dinov3-vitb16-pretrain-lvd1689m` | E1, frozen probe | Newest self-supervised features |
| P3 | MedSigLIP | `google/medsiglip-448` | E1, frozen probe | Medical foundation model |
| R1 | ResNet-50 | timm `resnet50.a1_in1k` | E2 | Standard CNN reference |
| R2 | DeiT-S | timm `deit_small_patch16_224.fb_in1k` | E2 | Quantization parent; base for B1 |
| R3 | DINOv3-B, fine-tuned | From P2 (timm `vit_base_patch16_dinov3.lvd1689m`, run at 224 px) | E2 | Quantization parent; strongest model; teacher for B2 |
| R4 | MobileNetV4-Conv-M | timm `mobilenetv4_conv_medium.e500_r256_in1k` | E2 | Quantization parent; efficient CNN |
| R5 | TinyViT-11M | timm `tiny_vit_11m_224.dist_in22k_ft_in1k` | E2 | Quantization parent; efficient hybrid ViT |
| B1 | BinaryAttention on DeiT-S | [EdwardChasel/BinaryAttention](https://github.com/EdwardChasel/BinaryAttention) | E4 | 1-bit attention |
| B2 | BHViT-Tiny, binary | [IMRL/BHViT](https://github.com/IMRL/BHViT), checkpoint listed as "binary 64.0" | E4 | Fully binary |
| B3 | BHViT-Tiny, full precision | Same repo, checkpoint listed as "somefp 66.0" — see note below | E4 | Reference for B2 |

All five timm tags were confirmed with `timm.list_pretrained` on 2026-11-01 (timm 1.0.30).

**Correction for B3.** The "somefp 66.0" checkpoint (`best66.0r.pth`) is **not** full precision: its saved arguments are `weight_bits=1, input_bits=1, some_fp=True`, i.e. binary blocks with full-precision intermediate patch-embedding layers. "binary 64.0" (`best64.0.pth`) is `some_fp=False`, fully binary. The repository publishes no 32-bit BHViT. B3 is therefore implemented as the BHViT-Tiny architecture at 32/32 bits, initialised from the latent real-valued weights of `best66.0r.pth` and fine-tuned with the E2 recipe; label it this way in the paper.

## E1: Frozen-feature baselines

E1 fits a logistic-regression probe on frozen embeddings from three foundation models and on colour histograms; it takes under an hour.

**Embeddings** (eval transform, no gradients, saved as float32 per split)

- **P1 DINOv2-B:** last layer's CLS token concatenated with the mean of its patch tokens.
- **P2 DINOv3-B:** the same, excluding register tokens (count from `config.num_register_tokens`). *Implementation:* timm `forward_features` output (after the final norm); its `num_prefix_tokens` = 1 CLS + 4 registers are skipped for the patch mean.
- **P3 MedSigLIP:** `get_image_features` output at 448 px.
- **C1:** HSV histogram, 32 bins per channel, normalized to sum to 1.

**Probe**

1. Fit `StandardScaler` on Train; apply it to every split.
2. `LogisticRegression(solver='lbfgs', max_iter=5000, class_weight={0: 1.855, 1: 0.624, 2: 1.165})`.
3. Search C over `np.logspace(-4, 2, 13)`; pick the best Val-A T1 AUROC, breaking ties by lower T2 log loss.
4. Fit on Train only, and write the `decision_function` outputs as logits.

## E2: Fine-tuning

E2 fine-tunes five networks with one shared recipe; four of them (R2–R5) are the full-precision parents that E3 compresses. About 20 runs, roughly 7–9 GPU-hours.

| Setting | Value |
| --- | --- |
| Head | `Linear(in_features, 3)` |
| Loss | Class-weighted cross-entropy, label smoothing 0.1 |
| Optimizer | AdamW, betas (0.9, 0.999), weight decay 0.05; none on biases and norm layers |
| Learning rate | R2–R5: search {1e-5, 3e-5, 1e-4} for transformers, {3e-5, 1e-4, 3e-4} for R4; seed 0 only. R1: 1e-4, no search |
| Layer-wise LR decay | 0.75 for R2 and R3 |
| Schedule | 30 epochs, 2-epoch linear warm-up, cosine decay to 0 |
| Early stopping | Val-A T1 AUROC, patience 7; keep the best checkpoint. *Implementation:* AUROC on untempered softmax (T = 1), since T is fitted afterwards |
| Batch | 32 (R3: 16 with 2-step gradient accumulation) |
| Precision | bf16 autocast; drop path 0.1 for transformers. *Implementation:* fp16 with gradient scaling on GPUs without bf16 (e.g. Colab T4) |
| Seeds | R2–R5: 0, 1, 2 at the chosen rate. R1: seed 0 |

**Run order:** R3 first (longest, 30–60 min per run), then R2, R5, R4, R1 (10–20 min each). Measure the first run and adjust the schedule.

## E3: Quantization

E3 compresses the four parents (R2–R5, all three seeds) without retraining: a simulated weight sweep at 8, 4, 2 and 1 bits, plus real INT8 models timed on CPU.

**E3a. Weight-only sweep (simulated, minutes per setting)**

1. Quantize the weights of every `nn.Linear` and `nn.Conv2d` except the first layer (stem or patch embedding) and the classifier, which stay full precision.
2. Per output channel, symmetric, uniform; clipping ratio c per channel chosen from {0.50, 0.55, …, 1.00} to minimize the squared weight error.
3. Bit widths b ∈ {8, 4, 2}; activations stay float32.
4. At b = 1, each channel becomes its sign times its mean absolute value.
5. Evaluate on every split; no retraining.

```latex
\Delta = \frac{c\,\max|w|}{2^{b-1}-1},\qquad \hat w = \Delta\cdot\mathrm{clamp}\big(\mathrm{round}(w/\Delta),\,-(2^{b-1}-1),\,2^{b-1}-1\big),\qquad \hat w_{\,b=1} = \overline{|w|}\cdot\mathrm{sign}(w)
```

**E3b. Real INT8 with ONNX Runtime**

1. Export to ONNX (opset 17, dynamic batch). *Implementation:* TorchScript exporter first, `torch.export` (dynamo) exporter as fallback; `quant_pre_process` (shape inference) before quantization.
2. `onnxruntime.quantization.quantize_static`: `QuantFormat.QDQ`, `per_channel=True`, INT8 weights and activations, MinMax calibration on 256 Train images (seed 0).
3. If Val-A T1 AUROC drops by more than 0.02, redo with percentile calibration and keep the better.
4. Models: R2, R4, R5; try R3 and record it if export fails.
5. Measure file size and CPU latency (protocol in Metrics).

## E4: Binary models

E4 trains the 1-bit end of the comparison: BinaryAttention on DeiT-S and a fully binary BHViT-Tiny, each with a 3-hour cap and a stop rule.

**B1. BinaryAttention (3 h cap)**

- [BinaryAttention](https://arxiv.org/abs/2603.09582) keeps only the sign of queries and keys in attention, adds a learnable bias, and trains with quantization-aware training plus self-distillation.
- Initialize from your fine-tuned DeiT-S (R2, best seed) and use it as the self-distillation teacher.
- Use the repo's training script and default fine-tuning settings, changing only: 3 classes, your splits, 30 epochs, early stopping on Val-A T1 AUROC (patience 7).
- *Implementation (`scripts/e4_binary.py binatt`):* the repo's `VisionTransformer` with its README flags `--attn-quant --attn-bias --pv-quant` (so P and V are also 8-bit; label the row "QK 1-bit, PV 8-bit"); weights from the best R2 seed (only the new relative-position bias tables start fresh); repo defaults lr 5e-5 → min 5e-6, weight decay 0.02, 5 warm-up epochs, hard distillation with α = 0.5 from the frozen R2 parent. The lr is used unscaled (the repo's batch/512 scaling would give 3e-6 at batch 32). The repo's mixup/cutmix/RandAugment/random erasing are replaced by the protocol augmentation ("no mixup or cutmix"). The base loss is the protocol's class-weighted CE with label smoothing 0.1.
- Report accuracy only; do not claim a speed-up you did not measure.

**B3. BHViT-Tiny, full precision (1 h)**

In `env-bhvit`, load the Tiny full-precision checkpoint from the README's Google Drive link, replace the head with 3 classes, and fine-tune with the E2 recipe (learning rate searched over {1e-4, 3e-4}). *Implementation:* see the B3 correction under Models; no layer-wise LR decay (hierarchical model).

**B2. BHViT-Tiny, fully binary (3 h cap)**

1. Load the Tiny binary checkpoint (64.0% ImageNet top-1) and replace the head with 3 classes.
2. Loss: 0.5 × class-weighted cross-entropy + 0.5 × T² × KL to R3's softened outputs (T = 4), plus the repo's oscillation regularization loss as set in its configs. *Implementation:* teacher = best R3 seed on Val-A; regulariser = mean over binary weight tensors (`cov1-3`, `dense`, `query`, `key`, `value`) of mean(| |w| − 1 |), weight 0.1 as in the repo's `train_one_epoch_L1`; KL with `batchmean`.
3. The repo's optimizer settings; if none are given for fine-tuning, AdamW at 1e-4 with cosine decay. *Implementation:* the repo gives ImageNet settings only, so AdamW 1e-4, cosine, no warm-up, weight decay 0 (the repo's value).
4. Up to 100 epochs, early stopping on Val-A T1 AUROC with patience 15.
5. **Stop rule:** if Val-A T1 AUROC is below 0.60 after 10 epochs, retry once at 0.3× the learning rate; if still below, stop and report the failure.

**Cost of binary models:** parameters, packed model size, and OPs = FLOPs + BOPs / 64 (the convention in binary-network papers), labelled as estimated. *Implementation (`tzq/cost.py`):* forward hooks count multiply-accumulates of every Linear/Conv layer plus the attention matmuls (2 N²C per block); a layer counts as BOPs when both its weights and inputs are 1-bit; FLOPs are counted as MACs, as in those papers.

## E5: Calibration and referral threshold

E5 sets a conformal referral threshold that bounds the share of TZ3 cases called eligible, then tests whether quantized and binary models still respect it when they inherit their parent's calibration. This is the paper's main result.

1. **Temperature scaling.** For every model and variant, fit one temperature T on Val-A logits by minimizing negative log-likelihood (L-BFGS, start 1.5, T in \[0.05, 20\]).
2. **Score.** s = softmax(z / T) for TZ3.
3. **Threshold on Val-B.** Sort the scores of the m TZ3 images in Val-B ascending; k = floor(α (m + 1)); τ = the k-th smallest. Eligible only if s < τ. Primary α = 0.10 (k = 12 when m ≈ 123); also report α = 0.05.

```latex
P(\text{eligible} \mid \text{TZ3}) = P(s_{\text{new}} < s_{(k)}) \le \frac{k}{m+1} \le \alpha
```

4. **Two regimes for every compressed variant (E3, E4)**

| Regime | Temperature and threshold |
| --- | --- |
| Inherited | The full-precision parent's T and τ, reused unchanged |
| Recalibrated | T refit on Val-A and τ refit on Val-B from the compressed model's own outputs |

5. **Report on Test with bootstrap CIs:** the missed-TZ3 rate (share of the 275 TZ3 images called eligible; should be ≤ α) and the eligibility rate (share of the 577 TZ1/2 images called eligible; higher is better).
6. **Outcomes per 1,000 examinations** for the best full-precision, INT8 and binary models: eligible (of which TZ3) and referred (of which TZ1/2), at Test prevalence (32% TZ3).

**Expected pattern to test:** the inherited regime misses more TZ3 cases as bit width falls, and recalibration brings the miss rate back under α at some cost in eligibility. If that holds, the conclusion is that compressed models must be recalibrated, not only re-validated for accuracy.

## Metrics and statistics

Six accuracy and calibration metrics, three cost measures, and stratified bootstrap CIs on the 852 test images; thresholds come only from Val-A or Val-B.

| Metric | Definition |
| --- | --- |
| T1 AUROC (primary) | `roc_auc_score(y == 2, s)` |
| TZ3 sensitivity at specificity 0.80 | Threshold where Val-A specificity first reaches 0.80, applied unchanged to Test. *Implementation:* on tempered s; cut = the ⌈0.8 n⌉-th smallest Val-A TZ1/2 score, TZ3 called when s > cut |
| T1 ECE | 15 equal-width bins on s, before and after temperature scaling |
| T2 macro-F1 | `f1_score(y, argmax, average='macro')` |
| Missed-TZ3 rate | Share of TZ3 test images with s < τ (E5) |
| Eligibility rate | Share of TZ1/2 test images with s < τ (E5) |

| Cost measure | How |
| --- | --- |
| Parameters and model size (MB) | Real models: file size on disk. Simulated: (b × quantized parameters + 32 × full-precision parameters) / 8 / 10⁶ |
| Compute (GMACs) | fvcore at native input size, batch 1; binary models as OPs, labelled estimated |
| CPU latency (ms) | ONNX Runtime, batch 1, 4 threads, 50 warm-up runs, then median of 200; FP32 and INT8 on the same machine; record the CPU model. *Implementation:* measured for seed 0 only (the architecture is the same across seeds) |

**Statistics**

1. Bootstrap: 2,000 resamples of the test images, stratified by label; percentile 95% CIs.
2. Models with 3 seeds: average over seeds inside each resample, and report the SD across seeds.
3. Compressed minus parent, computed on the same resamples. A variant is non-inferior if the lower 95% bound of its AUROC difference is above −0.03. Fix this margin before opening Test.

## Paper layout

Six IEEE two-column pages hold two tables and two figures; everything else goes to a short supplement or the code repository.

| Section | Pages | Contents |
| --- | --- | --- |
| I. Introduction | 0.75 | TZ types and ablation eligibility, why models must be compressed, contributions |
| II. Related work | 0.5 | TZ classification; quantized and binary vision transformers; calibration and conformal thresholds |
| III. Materials and methods | 1.5 | Data and split, models, quantization, binary models, calibration and threshold |
| IV. Results | 2.0 | Table I, Fig. 1, Fig. 2, Table II |
| V. Discussion and conclusion | 0.5 | Main finding, limitations |
| References | 0.75 | About 20 references |

**Exhibits**

| Exhibit | Content |
| --- | --- |
| Table I | Main results, one row per model (template below) |
| Fig. 1 | T1 AUROC vs. bit width (32, 8, 4, 2, 1) for R2–R5, with the INT8, BinaryAttention and BHViT results as marked points |
| Fig. 2 | Missed-TZ3 rate vs. bit width, inherited vs. recalibrated, with the α = 0.10 line |
| Table II | Outcomes per 1,000 examinations for the best full-precision, INT8 and binary models (or two sentences if space runs out) |

**Table I template**

| ID | Model | Bits (W/A) | Size (MB) | T1 AUROC \[95% CI\] | TZ3 sens. @ spec 0.80 | T1 ECE | T2 macro-F1 | CPU ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | HSV + LR | 32/32 |  |  |  |  |  |  |
| P1 | DINOv2-B probe | 32/32 |  |  |  |  |  |  |
| P2 | DINOv3-B probe | 32/32 |  |  |  |  |  |  |
| P3 | MedSigLIP probe | 32/32 |  |  |  |  |  |  |
| R1 | ResNet-50 | 32/32 |  |  |  |  |  |  |
| R2 | DeiT-S | 32/32 |  |  |  |  |  |  |
| R2 | DeiT-S | 8/8 |  |  |  |  |  |  |
| R3 | DINOv3-B fine-tuned | 32/32 |  |  |  |  |  |  |
| R4 | MobileNetV4-Conv-M | 32/32 |  |  |  |  |  |  |
| R4 | MobileNetV4-Conv-M | 8/8 |  |  |  |  |  |  |
| R5 | TinyViT-11M | 32/32 |  |  |  |  |  |  |
| R5 | TinyViT-11M | 8/8 |  |  |  |  |  |  |
| B1 | BinaryAttention (DeiT-S) | 32/32, QK 1-bit |  |  |  |  |  |  |
| B3 | BHViT-Tiny | 32/32 |  |  |  |  |  |  |
| B2 | BHViT-Tiny | 1/1 |  |  |  |  |  |  |

The 4-, 2- and 1-bit simulated results appear in Fig. 1, not as table rows.

## Writing notes

Build the paper around the E5 result, and fill the bracketed numbers from Table I and Fig. 2.

**Abstract skeleton (five sentences)**

1. Ablation eligibility in cervical screening depends on the transformation-zone type, and models that assess it on low-cost devices must be small.
2. We compare full-precision, quantized and binary vision transformers for TZ classification on 8,188 MobileODT images.
3. The best full-precision model reaches a TZ3-vs-rest AUROC of \[X\]; INT8 models stay within \[Y\]; a fully binary transformer reaches \[Z\].
4. Reusing the full-precision calibration after compression raises the missed-TZ3 rate from \[a\] to \[b\]; recalibration restores it below the target.
5. Compressed models should therefore be recalibrated, not only re-validated for accuracy.

**Fixes from the earlier draft**

- Use IFCPC TZ types throughout; never "three types of cervical cancer".
- GLOBOCAN 2022: about 662,000 new cases and 349,000 deaths (approximate, from memory; verify at gco.iarc.fr).
- No "ensemble" or "diagnosis" in the title; state that the paper makes no diagnostic claims.

**Limitations to state:** one dataset without external validation; possible train–test near-duplicates; manual crops; noisy labels in the additional set; CPU rather than on-phone timing; binary speed-ups estimated; the threshold guarantee assumes test images are exchangeable with Val-B.

## Sources

- [WHO guidelines for thermal ablation: TZ types and eligibility](https://www.ncbi.nlm.nih.gov/books/NBK549186/)
- [Transformation zone types in low-resource settings (ecancer, 2023)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10898880/)
- [BHViT code and checkpoints](https://github.com/IMRL/BHViT) · [BHViT paper (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Gao_BHViT_Binarized_Hybrid_Vision_Transformer_CVPR_2025_paper.html)
- [BinaryAttention (arXiv 2603.09582, CVPR 2026)](https://arxiv.org/abs/2603.09582)
- [DINOv3 code and checkpoints](https://github.com/facebookresearch/dinov3)
- [MedSigLIP model card](https://developers.google.com/health-ai-developer-foundations/medsiglip/model-card)

Cited from memory, not link-checked here: Guo et al. (2017) for temperature scaling; Angelopoulos and Bates for split conformal prediction.

"""Constants fixed by the experiment protocol."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("TZQ_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.environ.get("TZQ_DATA", ROOT / "data"))
SPLITS_DIR = ROOT / "splits"
FEATURES_DIR = ROOT / "features"
RUNS_DIR = ROOT / "runs"
QUANT_DIR = ROOT / "quant"
ANALYSIS_DIR = ROOT / "analysis"

HF_DATASET = "mach-12/cervical_cancer_classification"
CLASSES = ["Type_1", "Type_2", "Type_3"]  # -> 0, 1, 2
TZ3 = 2
CLASS_WEIGHTS = [1.855, 0.624, 1.165]  # N / (3 n_c) on Train
SPLITS = ["train", "valA", "valB", "test"]

# Pre-resized square cache (longest needed side is MedSigLIP's 448 px).
CACHE_SIZE = 512

# Fine-tuned models (E2). `size` overrides the timm config; MobileNetV4 keeps its own (256 px).
MODELS = {
    "resnet50": dict(id="R1", kind="timm", name="resnet50.a1_in1k", transformer=False, size=224),
    "deit_s": dict(id="R2", kind="timm", name="deit_small_patch16_224.fb_in1k", transformer=True, size=224),
    # Same LVD-1689M weights as facebook/dinov3-vitb16-pretrain-lvd1689m, mirrored ungated by timm.
    "dinov3_b": dict(id="R3", kind="timm", name="vit_base_patch16_dinov3.lvd1689m", transformer=True, size=224),
    "mnv4_m": dict(id="R4", kind="timm", name="mobilenetv4_conv_medium.e500_r256_in1k", transformer=False),
    "tinyvit_11m": dict(id="R5", kind="timm", name="tiny_vit_11m_224.dist_in22k_ft_in1k", transformer=True, size=224),
}

LR_GRID = {
    "resnet50": [1e-4],
    "deit_s": [1e-5, 3e-5, 1e-4],
    "dinov3_b": [1e-5, 3e-5, 1e-4],
    "mnv4_m": [3e-5, 1e-4, 3e-4],
    "tinyvit_11m": [1e-5, 3e-5, 1e-4],
}
LLRD = {"deit_s": 0.75, "dinov3_b": 0.75}
# Frozen feature extractors (E1).
PROBES = {
    "dinov2_b": dict(id="P1", kind="hf_dinov2", name="facebook/dinov2-base", size=224),
    "dinov3_b": dict(id="P2", kind="timm_tokens", name="vit_base_patch16_dinov3.lvd1689m", size=224),
    "medsiglip": dict(id="P3", kind="hf_siglip", name="google/medsiglip-448", size=448),
}

PARENTS = ["deit_s", "dinov3_b", "mnv4_m", "tinyvit_11m"]

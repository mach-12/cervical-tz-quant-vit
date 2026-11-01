"""GMACs (fvcore, batch 1, native input size) for the full-precision fine-tuned models.

    python scripts/cost_fp.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tzq.config import ANALYSIS_DIR, MODELS  # noqa: E402
from tzq.cost import estimate_cost, gmacs_fvcore  # noqa: E402
from tzq.models import build_classifier  # noqa: E402
from tzq.utils import save_json  # noqa: E402

if __name__ == "__main__":
    out = {}
    for key in MODELS:
        model, size, *_ = build_classifier(key, pretrained=False)
        out[key] = dict(size=size, gmacs_fvcore=gmacs_fvcore(model, size),
                        gmacs_hooks=estimate_cost(model, size)["flops_g"],
                        params_m=sum(p.numel() for p in model.parameters()) / 1e6)
        print(key, out[key])
    save_json(ANALYSIS_DIR / "gmacs.json", out)

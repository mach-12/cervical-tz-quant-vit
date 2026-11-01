"""Model builders for fine-tuning (E2) and frozen feature extraction (E1)."""
import timm
import torch
import torch.nn as nn

from .config import MODELS, PROBES


def build_classifier(key: str, pretrained: bool = True):
    """Return (model, size, mean, std) for a fine-tuned model in MODELS."""
    cfg = MODELS[key]
    kwargs = dict(drop_path_rate=0.1) if cfg["transformer"] else {}
    model = timm.create_model(cfg["name"], pretrained=pretrained, num_classes=3, **kwargs)
    dc = timm.data.resolve_model_data_config(model)
    size = cfg.get("size") or dc["input_size"][-1]
    return model, size, dc["mean"], dc["std"]


def first_and_classifier(model: nn.Module):
    """Names of the first conv/linear (stem or patch embedding) and the classifier head.

    These stay full precision in every quantized variant.
    """
    first = next(n for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear)))
    head = model.get_classifier()
    head_name = next(n for n, m in model.named_modules() if m is head)
    return first, head_name


class FeatureExtractor(nn.Module):
    """Frozen embedding for a probe in PROBES; forward returns a (B, D) float tensor."""

    def __init__(self, key: str):
        super().__init__()
        cfg = PROBES[key]
        self.kind, self.size = cfg["kind"], cfg["size"]
        if self.kind == "hf_dinov2":
            from transformers import AutoModel
            self.net = AutoModel.from_pretrained(cfg["name"])
            self.mean, self.std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        elif self.kind == "timm_tokens":
            self.net = timm.create_model(cfg["name"], pretrained=True, num_classes=0)
            dc = timm.data.resolve_model_data_config(self.net)
            self.mean, self.std = dc["mean"], dc["std"]
        elif self.kind == "hf_siglip":
            from transformers import AutoImageProcessor, AutoModel
            self.net = AutoModel.from_pretrained(cfg["name"])
            proc = AutoImageProcessor.from_pretrained(cfg["name"])
            self.mean, self.std = tuple(proc.image_mean), tuple(proc.image_std)
        else:
            raise ValueError(self.kind)
        self.net.eval()

    @torch.no_grad()
    def forward(self, x):
        if self.kind == "hf_dinov2":
            h = self.net(pixel_values=x).last_hidden_state  # CLS + patches, no registers in DINOv2
            return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)
        if self.kind == "timm_tokens":
            h = self.net.forward_features(x)  # CLS, register tokens, patches (after final norm)
            k = self.net.num_prefix_tokens  # 1 CLS + 4 registers
            return torch.cat([h[:, 0], h[:, k:].mean(1)], dim=1)
        out = self.net.get_image_features(pixel_values=x)
        return out if torch.is_tensor(out) else out.pooler_output

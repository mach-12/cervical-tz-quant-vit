"""E3a: simulated per-channel weight quantization (no retraining)."""
import torch
import torch.nn as nn

from .models import first_and_classifier

CLIP_GRID = torch.arange(50, 101, 5).float() / 100  # 0.50, 0.55, ..., 1.00


@torch.no_grad()
def quantize_weight(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric uniform per-output-channel quantization, MSE-optimal clip ratio per channel.

    bits == 1: each channel becomes mean|w| * sign(w).
    """
    flat = w.reshape(w.shape[0], -1).float()
    if bits == 1:
        return (flat.abs().mean(1, keepdim=True) * flat.sign()).reshape(w.shape).to(w.dtype)
    qmax = 2 ** (bits - 1) - 1
    wmax = flat.abs().amax(1, keepdim=True).clamp_min(1e-12)
    best_err = torch.full((flat.shape[0],), float("inf"), device=w.device)
    best = flat.clone()
    for c in CLIP_GRID.tolist():
        delta = c * wmax / qmax
        q = delta * torch.clamp(torch.round(flat / delta), -qmax, qmax)
        err = (q - flat).pow(2).sum(1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best[better] = q[better]
    return best.reshape(w.shape).to(w.dtype)


def quantize_model(model: nn.Module, bits: int):
    """Quantize every Linear/Conv2d weight in place except the first layer and the classifier.

    Returns parameter counts used for the simulated model size.
    """
    first, head = first_and_classifier(model)
    n_q = 0
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)) and name not in (first, head):
            m.weight.data = quantize_weight(m.weight.data, bits)
            n_q += m.weight.numel()
    n_total = sum(p.numel() for p in model.parameters())
    size_mb = (bits * n_q + 32 * (n_total - n_q)) / 8 / 1e6
    return dict(bits=bits, params_total=n_total, params_quantized=n_q, size_mb_simulated=size_mb,
                kept_fp=[first, head])

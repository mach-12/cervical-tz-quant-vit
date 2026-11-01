"""Compute and size estimates: fvcore GMACs for full-precision models, OPs = FLOPs + BOPs / 64 for binary ones."""
import torch
import torch.nn as nn


def _is_binary(m):
    return getattr(m, "weight_bits", 32) == 1 and getattr(m, "input_bits", 32) == 1


@torch.no_grad()
def estimate_cost(model: nn.Module, size: int, attn_qk_bits: int = 32):
    """Hook-based MAC count at batch 1. Binary Linear/Conv layers count as BOPs.

    Attention matmuls (QK^T and PV, 2 N^2 C MACs per block) are counted on modules whose class name
    ends in "Attention"; QK^T is binary when attn_qk_bits == 1 or the module binarizes its inputs.
    """
    macs = dict(fp=0, bin=0)
    hooks = []

    def linear_conv(m, inp, out):
        if isinstance(m, nn.Conv2d):
            n = out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        else:
            n = out.numel() * m.in_features
        macs["bin" if _is_binary(m) else "fp"] += n

    def attention(m, inp, out):
        x = inp[0] if inp else None
        if x is None or x.dim() != 3:
            return
        n, c = x.shape[1], x.shape[2]
        qk_binary = attn_qk_bits == 1 or getattr(m, "input_bits", 32) == 1
        macs["bin" if qk_binary else "fp"] += n * n * c  # QK^T
        macs["fp"] += n * n * c  # PV

    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            hooks.append(m.register_forward_hook(linear_conv))
        elif type(m).__name__ == "Attention":
            hooks.append(m.register_forward_hook(attention))
    was_training = model.training
    model.eval()
    p = next(model.parameters())
    model(torch.zeros(1, 3, size, size, device=p.device, dtype=p.dtype))
    model.train(was_training)
    for h in hooks:
        h.remove()

    n_bin = sum(m.weight.numel() for m in model.modules()
                if isinstance(m, (nn.Linear, nn.Conv2d)) and _is_binary(m))
    n_total = sum(p.numel() for p in model.parameters())
    return dict(params_total=n_total, params_binary=n_bin,
                size_mb_packed=(n_bin + 32 * (n_total - n_bin)) / 8 / 1e6,
                flops_g=macs["fp"] / 1e9, bops_g=macs["bin"] / 1e9,
                ops_g_estimated=(macs["fp"] + macs["bin"] / 64) / 1e9,
                note="estimated: hook count of Linear/Conv layers plus attention matmuls; FLOPs counted as "
                     "multiply-accumulates as in binary-network papers; OPs = FLOPs + BOPs/64")


def gmacs_fvcore(model: nn.Module, size: int):
    from fvcore.nn import FlopCountAnalysis
    model.eval()
    fca = FlopCountAnalysis(model, torch.zeros(1, 3, size, size))
    fca.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
    return fca.total() / 1e9  # fvcore counts one fused multiply-add as one flop, i.e. MACs

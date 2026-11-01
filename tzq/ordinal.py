"""Rank-consistent ordinal head for TZ1 < TZ2 < TZ3 (E6).

One score f(x) and ordered thresholds b_1 < b_2:  P(TZ >= k+1 | x) = sigmoid(f(x) - b_k).
The head returns log class probabilities, so the rest of the pipeline (softmax, temperature, AUROC,
quantization, ONNX) treats them as ordinary logits: softmax(log p) = p, and p(TZ3) = sigmoid(f - b_2)
is monotone in the single score f.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalHead(nn.Linear):
    """A Linear(in, 1) score plus ordered thresholds. Subclassing nn.Linear keeps it recognised as the
    classifier (it stays full precision in E3a, like every other head)."""

    def __init__(self, in_features: int, n_classes: int = 3):
        super().__init__(in_features, 1, bias=False)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.gaps = nn.Parameter(torch.zeros(n_classes - 2))  # b_{k+1} = b_k + softplus(gap_k) > b_k
        self.cum = None

    def thresholds(self):
        return torch.cat([self.b0, self.b0 + torch.cumsum(F.softplus(self.gaps), 0)])

    def init_thresholds(self, labels: torch.Tensor):
        """Start the thresholds at the training-set cumulative log-odds (f = 0 gives the class priors)."""
        n = len(labels)
        cum = torch.stack([(labels > k).float().sum() / n for k in range(len(self.gaps) + 1)])
        b = -torch.logit(cum.clamp(1e-3, 1 - 1e-3))
        with torch.no_grad():
            self.b0.copy_(b[:1])
            self.gaps.copy_(torch.log(torch.expm1((b[1:] - b[:-1]).clamp_min(1e-3))))  # inverse softplus

    def forward(self, x):
        cum = super().forward(x).float() - self.thresholds().float()  # (B, K-1) cumulative logits
        if self.training:
            self.cum = cum  # read by ordinal_loss; not stored in eval, so export/tracing see a pure function
        c = torch.sigmoid(cum)
        upper = torch.cat([torch.ones_like(c[:, :1]), c], 1)
        lower = torch.cat([c, torch.zeros_like(c[:, :1])], 1)
        return (upper - lower).clamp_min(1e-6).log()


def ordinal_loss(cum, y, class_weight, smoothing=0.1):
    """Class-weighted binary cross-entropy on the cumulative targets 1[y > k], with label smoothing.

    Weighted mean over images, as nn.CrossEntropyLoss(weight=...) does.
    """
    t = (y[:, None] > torch.arange(cum.shape[1], device=y.device)).float()
    t = t * (1 - smoothing) + smoothing / 2
    per_image = F.binary_cross_entropy_with_logits(cum, t, reduction="none").sum(1)
    w = class_weight[y]
    return (w * per_image).sum() / w.sum()

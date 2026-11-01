"""Accuracy, calibration, temperature scaling, and the conformal referral threshold."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax
from sklearn.metrics import f1_score, roc_auc_score

from .config import TZ3


def fit_temperature(logits, y):
    """One temperature on Val-A by NLL (L-BFGS, start 1.5, T in [0.05, 20])."""
    y = np.asarray(y)

    def nll(t):
        return -log_softmax(logits / t[0], axis=1)[np.arange(len(y)), y].mean()

    res = minimize(nll, x0=[1.5], method="L-BFGS-B", bounds=[(0.05, 20.0)])
    return float(res.x[0])


def tz3_score(logits, T=1.0):
    return softmax(np.asarray(logits) / T, axis=1)[:, TZ3]


def t1_auroc(y, s):
    return roc_auc_score(np.asarray(y) == TZ3, s)


def spec_threshold(y_ref, s_ref, spec=0.80):
    """Smallest cut at which specificity on the reference split first reaches `spec`.

    A case is called TZ3 when s > cut, so specificity = share of TZ1/2 scores <= cut.
    """
    neg = np.sort(np.asarray(s_ref)[np.asarray(y_ref) != TZ3])
    k = int(np.ceil(spec * len(neg)))
    return neg[k - 1]


def sensitivity(y, s, cut):
    return float((np.asarray(s)[np.asarray(y) == TZ3] > cut).mean())


def ece(y, s, bins=15):
    """T1 expected calibration error: 15 equal-width bins on s = p(TZ3)."""
    y = (np.asarray(y) == TZ3).astype(float)
    s = np.asarray(s)
    idx = np.minimum((s * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - s[m].mean())
    return float(total)


def macro_f1(y, logits):
    return f1_score(y, np.asarray(logits).argmax(1), average="macro")


def conformal_tau(y_valB, s_valB, alpha):
    """k-th smallest TZ3 score on Val-B with k = floor(alpha (m + 1)); eligible iff s < tau."""
    pos = np.sort(np.asarray(s_valB)[np.asarray(y_valB) == TZ3])
    k = int(np.floor(alpha * (len(pos) + 1)))
    return -np.inf if k < 1 else float(pos[k - 1])


def referral_rates(y, s, tau):
    y, s = np.asarray(y), np.asarray(s)
    eligible = s < tau
    return dict(missed_tz3=float(eligible[y == TZ3].mean()),
                eligibility=float(eligible[y != TZ3].mean()))


def stratified_bootstrap_indices(y, n_boot=2000, seed=0):
    """Resample indices within each label; the same indices are reused for every model (paired)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    groups = [np.flatnonzero(y == c) for c in np.unique(y)]
    return [np.concatenate([rng.choice(g, len(g), replace=True) for g in groups]) for _ in range(n_boot)]

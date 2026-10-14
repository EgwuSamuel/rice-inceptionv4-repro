"""Classification metrics, calibration and temperature scaling."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


def softmax(logits, temperature=1.0):
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(probs, y, n_bins=15) -> float:
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def classification_metrics(y, pred, labels, probs=None) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        y, pred, labels=labels, average="macro", zero_division=0
    )
    _, _, f1_c, support = precision_recall_fscore_support(
        y, pred, labels=labels, average=None, zero_division=0
    )
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
        "mcc": float(matthews_corrcoef(y, pred)),
        "kappa": float(cohen_kappa_score(y, pred, labels=labels)),
    }
    for lab, f, s in zip(labels, f1_c, support):
        out[f"f1_class{lab}"] = float(f)
        out[f"support_class{lab}"] = int(s)
    if probs is not None:
        out["ece"] = expected_calibration_error(probs, y)
    return out


def confusion(y, pred, n_classes):
    return confusion_matrix(y, pred, labels=list(range(n_classes)))


def fit_temperature(logits, y, max_iter=200) -> float:
    """Single-parameter temperature scaling fitted by NLL on validation logits."""
    logits_t = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    y_t = torch.as_tensor(np.asarray(y), dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits_t / log_t.exp(), y_t)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.exp().item())
    return t if np.isfinite(t) and t > 0 else 1.0

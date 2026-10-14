"""Statistical comparison: cluster bootstrap CIs, McNemar and Friedman tests."""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score


def cluster_bootstrap_ci(y, pred, groups, reps=2000, seed=0, alpha=0.05):
    """95% CI of accuracy and macro-F1, resampling source photographs (clusters).

    Augmented copies of one photograph are not independent, so resampling
    images would understate the uncertainty; whole groups are resampled instead.
    """
    y, pred, groups = map(np.asarray, (y, pred, groups))
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    labels = np.unique(y)
    rng = np.random.default_rng(seed)
    accs, f1s = [], []
    for _ in range(reps):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        accs.append((y[idx] == pred[idx]).mean())
        f1s.append(f1_score(y[idx], pred[idx], labels=labels, average="macro", zero_division=0))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "acc_ci_low": float(np.percentile(accs, lo)),
        "acc_ci_high": float(np.percentile(accs, hi)),
        "f1_ci_low": float(np.percentile(f1s, lo)),
        "f1_ci_high": float(np.percentile(f1s, hi)),
    }


def mcnemar_exact(correct_a, correct_b) -> dict:
    """Exact (binomial) McNemar test on paired per-image correctness."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01 = int((a & ~b).sum())  # A right, B wrong
    n10 = int((~a & b).sum())  # A wrong, B right
    n = n01 + n10
    p = 1.0 if n == 0 else float(stats.binomtest(n01, n, 0.5).pvalue)
    return {"a_only_correct": n01, "b_only_correct": n10, "p_value": p}


def bonferroni(pvalues):
    """Bonferroni-adjusted p-values (same order as input)."""
    m = len(pvalues)
    return [min(1.0, float(p) * m) for p in pvalues]


def friedman(scores_by_model: dict) -> dict:
    """Friedman test over folds; ``scores_by_model`` maps model -> per-fold scores."""
    arrays = [np.asarray(v, float) for v in scores_by_model.values()]
    if len(arrays) < 3 or len(arrays[0]) < 2:
        return {"statistic": float("nan"), "p_value": float("nan")}
    res = stats.friedmanchisquare(*arrays)
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue)}

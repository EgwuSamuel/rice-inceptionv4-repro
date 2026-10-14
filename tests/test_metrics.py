import numpy as np

from ricedx import evaluate as E
from ricedx import stats as S


def test_perfect_predictions():
    y = np.array([0, 1, 2, 3, 0, 1])
    m = E.classification_metrics(y, y, [0, 1, 2, 3])
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0 and m["mcc"] == 1.0


def test_known_macro_f1():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 1, 1, 1])
    m = E.classification_metrics(y, p, [0, 1])
    # class 0: P=1, R=0.5, F1=0.667; class 1: P=0.667, R=1, F1=0.8
    assert abs(m["macro_f1"] - (2 / 3 + 0.8) / 2) < 1e-9


def test_ece_zero_for_calibrated_onehot():
    y = np.array([0, 1, 2])
    assert E.expected_calibration_error(np.eye(3), y) == 0.0


def test_temperature_scaling_reduces_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400)
    logits = rng.normal(0, 1, (400, 4))
    logits[np.arange(400), y] += 1.0
    logits *= 6  # over-confident
    t = E.fit_temperature(logits, y)
    assert t > 1.5
    assert E.expected_calibration_error(E.softmax(logits, t), y) < E.expected_calibration_error(
        E.softmax(logits), y
    )


def test_mcnemar_and_bonferroni():
    a = np.array([1] * 30 + [0] * 5, bool)
    b = np.array([0] * 20 + [1] * 15, bool)
    r = S.mcnemar_exact(a, b)
    assert r["a_only_correct"] == 20 and r["b_only_correct"] == 5 and r["p_value"] < 0.01
    assert np.allclose(S.bonferroni([0.01, 0.04, 0.5]), [0.03, 0.12, 1.0])


def test_cluster_bootstrap_bounds():
    y = np.repeat([0, 1], 50)
    pred = y.copy()
    pred[:10] = 1
    groups = np.repeat(np.arange(20), 5)
    ci = S.cluster_bootstrap_ci(y, pred, groups, reps=200)
    assert ci["acc_ci_low"] <= 0.9 <= ci["acc_ci_high"]

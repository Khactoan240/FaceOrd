"""Evaluation metrics (fractions in [0, 1], not percentages).

Age: accuracy, macro F1, macro one-vs-rest AUC, mAP, multiclass Brier score, ECE
(10 bins), class MAE and adjacent accuracy. Gender: accuracy, F1, AUC, AP, Brier, ECE.
"""
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


def cumulative_diagnostics(cumulative, tolerance=0.0):
    """Monotonicity violations of the raw cumulative outputs c_k = P(y > k).

    c_k should be non-increasing in k. Reports the share of violated adjacent pairs,
    the share of samples with at least one violation and the mean violation
    magnitude over all pairs and over violated pairs.
    """
    c = np.asarray(cumulative, dtype=float)
    if c.ndim != 2 or c.shape[1] != 7 or not len(c) or not np.isfinite(c).all():
        raise ValueError('Expected nonempty, finite raw cumulative outputs with seven thresholds')
    if np.any(c < 0) or np.any(c > 1) or tolerance < 0:
        raise ValueError('Invalid cumulative probabilities or tolerance')
    magnitude = np.maximum(c[:, 1:] - c[:, :-1], 0)
    violated = magnitude > tolerance
    return {'n_samples': len(c), 'n_pairs': int(magnitude.size), 'tolerance': tolerance,
            'pairwise_violation_rate': float(violated.mean()),
            'sample_violation_rate': float(violated.any(axis=1).mean()),
            'mean_violation_magnitude_all_pairs': float(magnitude.mean()),
            'mean_violation_magnitude_violated_pairs': float(magnitude[violated].mean()) if violated.any() else 0.0}


def expected_calibration_error(y_true, probs, n_bins=10):
    """ECE with equal-width confidence bins; binary input is P(class 1)."""
    probs = np.asarray(probs)
    if probs.ndim == 1:
        confidence = np.maximum(probs, 1 - probs)
        pred = (probs >= 0.5).astype(int)
    else:
        confidence = probs.max(axis=1)
        pred = probs.argmax(axis=1)
    ids = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    return float(sum(
        np.mean(ids == b) * abs(np.mean(pred[ids == b] == np.asarray(y_true)[ids == b])
                                - np.mean(confidence[ids == b]))
        for b in range(n_bins) if np.any(ids == b)
    ))


def validate_probabilities(probs):
    """Check that predictions are finite probabilities and that multiclass rows sum to one."""
    p = np.asarray(probs, dtype=float)
    if p.size == 0 or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("Predictions must be finite probabilities in [0, 1]")
    if p.ndim == 2 and not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Age probability rows must sum to one")
    if p.ndim not in (1, 2):
        raise ValueError("Expected binary vector or multiclass matrix")
    return p


def compute_metrics(y_true, probs, binary=False):
    """Metrics of one task; AUC/AP of classes absent from y_true are skipped (None if undefined)."""
    y = np.asarray(y_true, dtype=int)
    p = validate_probabilities(probs)
    if len(y) != len(p):
        raise ValueError("Label and prediction lengths differ")
    pred = (p >= 0.5).astype(int) if binary else p.argmax(axis=1)
    result = {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, average="binary" if binary else "macro", zero_division=0)),
        "ece": expected_calibration_error(y, p),
    }
    if binary:
        result["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
        result["map"] = float(average_precision_score(y, p)) if np.any(y == 1) else None
        result["brier"] = float(np.mean((p - y) ** 2))
    else:
        one_hot = np.eye(p.shape[1])[y]
        per_class_auc, per_class_ap = {}, {}
        for k in range(p.shape[1]):
            target = one_hot[:, k]
            if 0 < target.sum() < len(y):
                per_class_auc[str(k)] = float(roc_auc_score(target, p[:, k]))
                per_class_ap[str(k)] = float(average_precision_score(target, p[:, k]))
        result["auc"] = float(np.mean(list(per_class_auc.values()))) if per_class_auc else None
        result["map"] = float(np.mean(list(per_class_ap.values()))) if per_class_ap else None
        result["auc_classes"] = list(per_class_auc)
        result["per_class_auc"] = per_class_auc
        result["brier"] = float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))
        result["mae_class"] = float(np.mean(np.abs(y - pred)))
        result["adj_acc"] = float(np.mean(np.abs(y - pred) <= 1))
    return result

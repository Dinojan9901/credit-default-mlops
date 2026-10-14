"""Metrics and decision-threshold selection."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

PRIMARY_METRIC = "roc_auc"


def choose_threshold(y_true: ArrayLike, proba: ArrayLike) -> float:
    """Threshold that maximises F1 on the given (validation) data."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision/recall have one more element than thresholds; drop the last point.
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(thresholds[int(np.argmax(f1))])


def classification_metrics(
    y_true: ArrayLike, proba: ArrayLike, threshold: float = 0.5
) -> dict[str, float]:
    y = np.asarray(y_true)
    p = np.asarray(proba, dtype=float)
    pred = (p >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
    }

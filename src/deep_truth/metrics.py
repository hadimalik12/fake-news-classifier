"""Metrics and hypothesis decisions from the approach spec.

Macro averages are unweighted across the label list in config.yaml.
Comparison uses four decimal places, and a tie does not support a hypothesis.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def round_metric(value: float, decimals: int) -> float:
    return round(float(value), decimals)


def strictly_greater(left: float, right: float, decimals: int) -> bool:
    return round_metric(left, decimals) > round_metric(right, decimals)


def compute_metrics(y_true, y_pred, labels: list[str]) -> dict:
    """Score string labels. Rows of the confusion matrix are the true labels."""
    per_precision, per_recall, per_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class_precision": {label: float(value) for label, value in zip(labels, per_precision)},
        "per_class_recall": {label: float(value) for label, value in zip(labels, per_recall)},
        "per_class_f1": {label: float(value) for label, value in zip(labels, per_f1)},
        "confusion_matrix": np.asarray(matrix).tolist(),
    }


def hypothesis_1_supported(full_f1: float, tfidf_f1: float, bilstm_f1: float, decimals: int) -> bool:
    return strictly_greater(full_f1, tfidf_f1, decimals) and strictly_greater(full_f1, bilstm_f1, decimals)


def hypothesis_2_supported(
    full_f1: Mapping[str, float],
    tfidf_f1: Mapping[str, float],
    bilstm_f1: Mapping[str, float],
    decimals: int,
) -> bool:
    tfidf_improves = strictly_greater(
        full_f1["long"] - tfidf_f1["long"],
        full_f1["short"] - tfidf_f1["short"],
        decimals,
    )
    bilstm_improves = strictly_greater(
        full_f1["long"] - bilstm_f1["long"],
        full_f1["short"] - bilstm_f1["short"],
        decimals,
    )
    return tfidf_improves and bilstm_improves


def hypothesis_3_supported(full_f1: float, frozen_f1: float, decimals: int) -> bool:
    return strictly_greater(full_f1, frozen_f1, decimals)

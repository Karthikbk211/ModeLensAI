"""
Core model evaluation metrics.

Given a fitted sklearn-compatible classifier and a held-out (X, y) set,
compute the standard classification metrics ModelLens reports on the
dashboard: accuracy, precision, recall, F1, ROC-AUC (when applicable),
confusion matrix and the full classification report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    """Container for everything downstream modules (failure analysis,
    GenAI, database) need from a single evaluation run."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    confusion: np.ndarray
    class_labels: list[Any]
    classification_report: dict
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray | None = field(default=None, repr=False)

    def to_summary_dict(self) -> dict:
        """A small, JSON-serialisable summary safe to store in Postgres
        or hand to the GenAI layer."""
        return {
            "accuracy": round(float(self.accuracy), 4),
            "precision": round(float(self.precision), 4),
            "recall": round(float(self.recall), 4),
            "f1_score": round(float(self.f1), 4),
            "roc_auc": round(float(self.roc_auc), 4) if self.roc_auc is not None else None,
            "class_labels": [str(c) for c in self.class_labels],
        }


def evaluate_model(model, X, y) -> EvaluationResult:
    """Run a fitted classifier against evaluation data and compute the
    full metrics bundle used throughout ModelLens.

    Parameters
    ----------
    model : a fitted scikit-learn-compatible classifier (must implement
        .predict, and .predict_proba if ROC-AUC is desired)
    X : array-like / DataFrame of features
    y : array-like / Series of true labels
    """
    y_true = np.asarray(y)
    y_pred = np.asarray(model.predict(X))

    class_labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    is_binary = len(class_labels) == 2

    average = "binary" if is_binary else "macro"
    # binary precision/recall/f1 need pos_label handling when labels aren't 0/1
    pos_label = class_labels[-1] if is_binary else 1

    kwargs = {"average": average, "zero_division": 0}
    if is_binary:
        kwargs["pos_label"] = pos_label

    precision = precision_score(y_true, y_pred, **kwargs)
    recall = recall_score(y_true, y_pred, **kwargs)
    f1 = f1_score(y_true, y_pred, **kwargs)
    accuracy = accuracy_score(y_true, y_pred)

    roc_auc = None
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = np.asarray(model.predict_proba(X))
            if is_binary:
                # label_binarize handles string labels safely
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(y_true, classes=class_labels).ravel()
                roc_auc = roc_auc_score(y_bin, y_proba[:, 1])
            else:
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(y_true, classes=class_labels)
                roc_auc = roc_auc_score(
                    y_bin, y_proba, multi_class="ovr", average="macro"
                )
        except Exception:
            # Some models expose predict_proba but fail on edge cases
            # (single-class fold, etc.) -- ROC-AUC is a nice-to-have.
            roc_auc = None

    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    report = classification_report(
        y_true, y_pred, labels=class_labels, output_dict=True, zero_division=0
    )

    return EvaluationResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        confusion=cm,
        class_labels=class_labels,
        classification_report=report,
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
    )

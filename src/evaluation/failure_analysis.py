"""
Failure analysis engine -- the core intelligence of ModelLens.

Given an EvaluationResult plus the original feature matrix, this module
answers three questions:

  1. Which classes does the model struggle with? (class-level failures)
  2. What does it confuse with what? (misclassification patterns)
  3. Are errors concentrated in particular feature ranges? (feature analysis)

The output is a plain dict, deliberately kept flat and JSON-serialisable
so it can be (a) stored in Postgres, (b) rendered in Streamlit, and
(c) handed directly to the GenAI layer as context.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.confusion import per_class_accuracy, top_misclassifications


def analyze_class_failures(confusion: np.ndarray, class_labels: list) -> list[dict]:
    """Flag classes whose recall is notably below the average recall
    across all classes."""
    acc = per_class_accuracy(confusion, class_labels)
    values = list(acc.values())
    mean_acc = float(np.mean(values)) if values else 0.0

    findings = []
    for label, class_acc in acc.items():
        gap = mean_acc - class_acc
        findings.append(
            {
                "class": label,
                "accuracy": class_acc,
                "gap_vs_average": round(gap, 4),
                "flagged": gap > 0.10,  # more than 10pp below average
            }
        )
    findings.sort(key=lambda f: f["accuracy"])
    return findings


def analyze_misclassifications(confusion: np.ndarray, class_labels: list, top_n: int = 5) -> list[dict]:
    return top_misclassifications(confusion, class_labels, top_n=top_n)


def analyze_feature_errors(
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_features: int = 8,
    bins: int = 4,
) -> list[dict]:
    """For each numeric feature, bucket samples into quantile bins and
    report the error rate per bucket. Surfaces feature ranges where the
    model is disproportionately wrong.

    Returns a list of {feature, bins: [{range, error_rate, n_samples}]}
    sorted by how much the worst bucket's error rate deviates from the
    feature's overall error rate (most informative features first).
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Reset index so numpy arrays and pandas Series align correctly
    # (e.g. after train_test_split which preserves original indices).
    X = X.reset_index(drop=True)
    is_error = pd.Series((np.asarray(y_true) != np.asarray(y_pred)).astype(int))
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    results = []
    for col in numeric_cols:
        series = X[col]
        if series.nunique() < bins:
            continue
        try:
            bucketed = pd.qcut(series, q=bins, duplicates="drop")
        except ValueError:
            continue

        df = pd.DataFrame({"bucket": bucketed, "is_error": is_error})
        grouped = df.groupby("bucket", observed=True)["is_error"].agg(["mean", "count"])

        overall_error_rate = float(is_error.mean())
        bucket_rows = []
        max_deviation = 0.0
        for interval, row in grouped.iterrows():
            error_rate = round(float(row["mean"]), 4)
            bucket_rows.append(
                {
                    "range": f"{interval.left:.2f} - {interval.right:.2f}",
                    "error_rate": error_rate,
                    "n_samples": int(row["count"]),
                }
            )
            max_deviation = max(max_deviation, abs(error_rate - overall_error_rate))

        results.append(
            {
                "feature": col,
                "overall_error_rate": round(overall_error_rate, 4),
                "bins": bucket_rows,
                "max_deviation": round(max_deviation, 4),
            }
        )

    results.sort(key=lambda r: r["max_deviation"], reverse=True)
    return results[:max_features]


def get_misclassified_samples(
    X: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, limit: int = 50
) -> pd.DataFrame:
    """Return the rows the model got wrong, with actual/predicted columns
    attached, for drill-down in the UI."""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    mask = np.asarray(y_true) != np.asarray(y_pred)
    out = X[mask].copy()
    out.insert(0, "actual", np.asarray(y_true)[mask])
    out.insert(1, "predicted", np.asarray(y_pred)[mask])
    return out.head(limit)


def run_failure_analysis(evaluation_result, X: pd.DataFrame) -> dict[str, Any]:
    """Top-level entry point: bundles all three analyses into one dict.

    Parameters
    ----------
    evaluation_result : src.evaluation.metrics.EvaluationResult
    X : the feature matrix used for evaluation (same rows as y_true/y_pred)
    """
    class_failures = analyze_class_failures(
        evaluation_result.confusion, evaluation_result.class_labels
    )
    misclassifications = analyze_misclassifications(
        evaluation_result.confusion, evaluation_result.class_labels
    )
    feature_errors = analyze_feature_errors(
        X, evaluation_result.y_true, evaluation_result.y_pred
    )

    return {
        "class_failures": class_failures,
        "top_misclassifications": misclassifications,
        "feature_errors": feature_errors,
    }

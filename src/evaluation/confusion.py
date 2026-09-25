"""
Confusion-matrix utilities: turning a raw matrix into per-class
accuracy figures and the most common misclassification pairs, plus a
ready-to-render Plotly heatmap.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def per_class_accuracy(confusion: np.ndarray, class_labels: list) -> dict:
    """Recall / accuracy for each class: diagonal / row sum."""
    result = {}
    for i, label in enumerate(class_labels):
        row_total = confusion[i].sum()
        correct = confusion[i, i]
        result[str(label)] = round(float(correct / row_total), 4) if row_total else 0.0
    return result


def top_misclassifications(confusion: np.ndarray, class_labels: list, top_n: int = 5) -> list[dict]:
    """Return the top-N (actual -> predicted) error pairs, excluding the
    diagonal (correct predictions)."""
    pairs = []
    for i, actual in enumerate(class_labels):
        for j, predicted in enumerate(class_labels):
            if i == j:
                continue
            count = int(confusion[i, j])
            if count > 0:
                pairs.append({"actual": str(actual), "predicted": str(predicted), "count": count})
    pairs.sort(key=lambda p: p["count"], reverse=True)
    return pairs[:top_n]


def confusion_matrix_figure(confusion: np.ndarray, class_labels: list) -> go.Figure:
    labels = [str(c) for c in class_labels]
    fig = go.Figure(
        data=go.Heatmap(
            z=confusion,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=confusion,
            texttemplate="%{text}",
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis_autorange="reversed",
        height=450,
    )
    return fig

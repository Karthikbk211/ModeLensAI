import pickle
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.confusion import per_class_accuracy, top_misclassifications
from src.evaluation.failure_analysis import run_failure_analysis
from src.evaluation.metrics import evaluate_model

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data"


@pytest.fixture(scope="module")
def sample_model_and_data():
    with open(SAMPLE_DIR / "sample_model.pkl", "rb") as f:
        model = pickle.load(f)
    df = pd.read_csv(SAMPLE_DIR / "sample_dataset.csv")
    X = df.drop(columns=["target"])
    y = df["target"]
    return model, X, y


def test_evaluate_model_returns_sane_metrics(sample_model_and_data):
    model, X, y = sample_model_and_data
    result = evaluate_model(model, X, y)

    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert 0.0 <= result.f1 <= 1.0
    assert result.roc_auc is not None
    assert result.confusion.shape == (len(result.class_labels), len(result.class_labels))


def test_per_class_accuracy_sums_correctly(sample_model_and_data):
    model, X, y = sample_model_and_data
    result = evaluate_model(model, X, y)
    acc = per_class_accuracy(result.confusion, result.class_labels)
    assert set(acc.keys()) == {str(c) for c in result.class_labels}
    for v in acc.values():
        assert 0.0 <= v <= 1.0


def test_top_misclassifications_excludes_diagonal(sample_model_and_data):
    model, X, y = sample_model_and_data
    result = evaluate_model(model, X, y)
    pairs = top_misclassifications(result.confusion, result.class_labels)
    for p in pairs:
        assert p["actual"] != p["predicted"]


def test_run_failure_analysis_shape(sample_model_and_data):
    model, X, y = sample_model_and_data
    result = evaluate_model(model, X, y)
    failure = run_failure_analysis(result, X)

    assert "class_failures" in failure
    assert "top_misclassifications" in failure
    assert "feature_errors" in failure
    assert isinstance(failure["feature_errors"], list)
    # every feature-error entry should have per-bucket error rates
    for fe in failure["feature_errors"]:
        assert "bins" in fe and len(fe["bins"]) > 0

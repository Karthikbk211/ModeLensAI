"""
Loading and light validation for uploaded model + dataset files.
"""

from __future__ import annotations

import io
import os
import pickle
from typing import Any

import pandas as pd


def get_config(key: str, default: str | None = None) -> str | None:
    """Read a config value from the environment first, falling back to
    Streamlit's secrets manager (used on Streamlit Community Cloud,
    where values are set in the app's Secrets panel / secrets.toml
    rather than as real env vars).
    """
    value = os.environ.get(key)
    if value:
        return value
    try:
        import streamlit as st

        return st.secrets.get(key, default)
    except Exception:
        return default


class ModelLensError(Exception):
    """Raised for user-facing problems (bad file, missing column, etc.)."""


def load_model(file_bytes: bytes) -> Any:
    """Unpickle an uploaded scikit-learn-compatible model.

    NOTE: unpickling executes arbitrary code. This is fine for a
    single-user local/demo tool, but if ModelLens is ever deployed
    multi-tenant, uploaded pickles must be sandboxed or replaced with a
    safer format (e.g. skops).
    """
    try:
        model = pickle.loads(file_bytes)
    except Exception as exc:  # noqa: BLE001
        raise ModelLensError(f"Could not load model file: {exc}") from exc

    if not hasattr(model, "predict"):
        raise ModelLensError("Uploaded object has no .predict method -- is this a fitted sklearn model?")
    return model


def load_dataset(file_bytes: bytes, filename: str) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    name_lower = filename.lower()
    if name_lower.endswith(".csv"):
        df = pd.read_csv(buf)
    elif name_lower.endswith(".parquet"):
        df = pd.read_parquet(buf)
    else:
        raise ModelLensError("Dataset must be a .csv or .parquet file")

    if df.empty:
        raise ModelLensError("Uploaded dataset is empty")
    return df


def split_features_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise ModelLensError(f"Target column '{target_column}' not found in dataset")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

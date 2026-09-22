"""Read, audit and validate the supplied dataset without learning from test data."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

TARGET = "Appliances"
LOCAL = ["lights"] + [item for i in range(1, 10) for item in (f"T{i}", f"RH_{i}")]


def load_data(path):
    """Parse numeric fields, sort time, and reject ambiguous or irregular timestamps.

    Missing sensor values are permitted (causal fill and training medians follow).
    Missing target labels and gaps are rejected rather than silently fabricated.
    """
    frame = pd.read_csv(path)
    required = ["date", TARGET, *LOCAL]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError("Missing or duplicate timestamps require a documented resolution.")
    frame = frame.sort_values("date").set_index("date")
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if np.isinf(frame.to_numpy()).any():
        raise ValueError("Infinite measurements must be investigated.")
    if frame[TARGET].isna().any() or (frame[TARGET] < 0).any():
        raise ValueError("Missing/negative target labels must be investigated, not imputed.")
    if not frame.index.to_series().diff().iloc[1:].eq(pd.Timedelta(minutes=10)).all():
        raise ValueError("Expected a regular 10-minute grid; do not bridge gaps silently.")
    return frame


def split_boundaries(n_rows):
    """First 80% is development; its first 80% is training (64/16/20 overall)."""
    development_end = int(n_rows * 0.80)
    train_end = int(development_end * 0.80)
    return train_end, development_end


def fit_feature_preprocessing(features, train_end, history_start):
    """Fit median imputation and standardization on complete-history training rows.

    Transform the full timeline with those fixed statistics. Validation and test
    rows never contribute to the fitted medians, means or standard deviations.
    """
    training = features.iloc[history_start:train_end]
    if training.empty or training.isna().all().any():
        raise ValueError("Every feature needs observed training values.")
    imputer = SimpleImputer(strategy="median").fit(training)
    scaler = StandardScaler().fit(imputer.transform(training))
    transformed = scaler.transform(imputer.transform(features)).astype(np.float32)
    return transformed, imputer, scaler


def audit_data(frame, source_path):
    """Return data-quality facts, provenance and outlier flags, without changing data."""
    train_end, test_start = split_boundaries(len(frame))
    train = frame.iloc[:train_end]
    q1, q3 = train[TARGET].quantile([0.25, 0.75])
    low, high = float(q1 - 1.5 * (q3 - q1)), float(q3 + 1.5 * (q3 - q1))
    flags = ((frame[TARGET] < low) | (frame[TARGET] > high))
    return {
        "source_sha256": hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
        "rows": len(frame), "columns_including_date": len(frame.columns) + 1,
        "start": str(frame.index[0]), "end": str(frame.index[-1]),
        "missing_cells": int(frame.isna().sum().sum()),
        "missing_by_column": frame.isna().sum().astype(int).to_dict(),
        "duplicate_timestamps": int(frame.index.duplicated().sum()),
        "interval_minutes": 10,
        "rv1_equals_rv2": bool(frame.rv1.equals(frame.rv2)) if "rv1" in frame and "rv2" in frame else None,
        "target_summary_wh": frame[TARGET].describe(percentiles=[0.01, 0.5, 0.95, 0.99]).to_dict(),
        "train_iqr_bounds_wh": [low, high],
        "iqr_flag_count_train": int(flags.iloc[:train_end].sum()),
        "iqr_flag_count_all": int(flags.sum()),
        "outlier_action": "Retain all observed energy peaks; an IQR flag is not proof of sensor error.",
        "train_end_exclusive": train_end, "test_start": test_start,
        "airport_weather_policy": "Exclude from predictors: upstream hourly interpolation is not demonstrably causal.",
    }

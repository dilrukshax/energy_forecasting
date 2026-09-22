"""Data loading, temporal validation, cleaning, and scaling routines."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TARGET = "Appliances"
FREQ = "10min"
NO_CLIP_FEATURES = {
    "hour", "day_of_week", "month", "is_weekend", "nsm",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_holiday", "is_non_working", "evening_peak_flag", "hour_x_weekend"
}


def load_data(path):
    """Load the raw CSV, parse timestamps, sort chronologically, and validate."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    frame = pd.read_csv(p, parse_dates=["date"]).sort_values("date").set_index("date")

    if frame.index.duplicated().any():
        raise ValueError("Dataset contains duplicate timestamps.")

    if TARGET not in frame.columns:
        raise ValueError(f"Target column '{TARGET}' missing from dataset.")

    if frame[TARGET].isna().any():
        raise ValueError("Missing target labels found; target cannot be imputed.")

    # Drop documented random noise variables if present
    drop_cols = [c for c in ["rv1", "rv2"] if c in frame.columns]
    if drop_cols:
        frame = frame.drop(columns=drop_cols)

    return frame


def enforce_regular_grid(frame, freq=FREQ, interp_limit=6):
    """Ensure time index is on a regular grid and causal forward-fill short gaps."""
    full_idx = pd.date_range(frame.index.min(), frame.index.max(), freq=freq)
    if len(full_idx) == len(frame) and full_idx.equals(frame.index):
        return frame.copy()

    reindexed = frame.reindex(full_idx)

    # Exogenous columns can be forward-filled for up to interp_limit steps (1 hour for 10-min data)
    exog = [c for c in reindexed.columns if c != TARGET]
    reindexed[exog] = reindexed[exog].ffill(limit=interp_limit)

    # Missing targets cannot be imputed
    if reindexed[TARGET].isna().any():
        raise ValueError(
            "Target column has missing timestamps on the regular grid that cannot be safely fabricated."
        )

    return reindexed


def iqr_bounds(series, k=1.5):
    """Compute Tukey IQR bounds (Q1 - k*IQR, Q3 + k*IQR)."""
    clean = series.dropna()
    q1, q3 = np.percentile(clean, [25, 75])
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def calculate_winsor_bounds(df, clip_cols=None, k=3.0):
    """Derive winsorization fences for predictor columns (computed on training data only)."""
    if clip_cols is None:
        clip_cols = [c for c in df.columns if c not in NO_CLIP_FEATURES]
    return {c: iqr_bounds(df[c], k=k) for c in clip_cols if c in df.columns}


def winsorise(frame, bounds):
    """Clip features to pre-computed bounds."""
    out = frame.copy()
    for col, (low, high) in bounds.items():
        if col in out.columns:
            out[col] = out[col].clip(low, high)
    return out


class TargetScaler:
    """StandardScaler in log1p space with exact inverse transformation back to Wh."""

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, y):
        values = np.log1p(np.asarray(y, float)).reshape(-1, 1)
        self.scaler.fit(values)
        return self

    def transform(self, y):
        values = np.log1p(np.asarray(y, float)).reshape(-1, 1)
        return self.scaler.transform(values).ravel()

    def inverse_transform(self, y_scaled):
        values = np.asarray(y_scaled, float).reshape(-1, 1)
        unscaled = self.scaler.inverse_transform(values).ravel()
        # Appliances energy consumption is strictly non-negative
        return np.maximum(0.0, np.expm1(unscaled))


def save_preprocessed_data(X_train, y_train, X_val, y_val, X_test, y_test, output_dir="data/processed"):
    """Save processed feature matrices and targets to CSV files."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for name, X_split, y_split in [("train", X_train, y_train),
                                   ("val", X_val, y_val),
                                   ("test", X_test, y_test)]:
        df_split = pd.DataFrame(X_split, index=y_split.index) if not isinstance(X_split, pd.DataFrame) else X_split.copy()
        df_split["target_Wh"] = y_split.values
        df_split.to_csv(out_path / f"{name}_processed.csv")

    print(f"Preprocessed data splits successfully saved to {out_path}")

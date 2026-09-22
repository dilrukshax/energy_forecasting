"""Evaluation metrics, always computed in the original units.

Scoring in transformed space flatters a model: an RMSE in standardised log space is not
comparable across transformations and cannot be explained to anyone. Every function here takes
and returns Wh.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class Metrics:
    """Regression metrics for one model on one split.

    Attributes:
        model: Name of the model.
        mae: Mean absolute error, Wh. Robust to the spikes, and the headline number.
        rmse: Root mean squared error, Wh. Penalises the big evening misses quadratically.
        mape: Mean absolute percentage error. Inflated by low-consumption night rows, so it is
            reported but not optimised.
        r2: Share of variance explained.
        n: Number of scored rows.
    """

    model: str
    mae: float
    rmse: float
    mape: float
    r2: float
    n: int

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


def compute_metrics(y_true: Iterable[float], y_pred: Iterable[float], model: str) -> Metrics:
    """Score predictions against actuals, in Wh.

    Args:
        y_true: Observed consumption.
        y_pred: Predicted consumption.
        model: Label recorded on the result.

    Returns:
        A :class:`Metrics` record.

    Raises:
        ValueError: if the inputs have different lengths.
    """
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {actual.shape} vs {predicted.shape}")

    nonzero = actual != 0
    mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)

    return Metrics(
        model=model,
        mae=float(mean_absolute_error(actual, predicted)),
        rmse=float(np.sqrt(mean_squared_error(actual, predicted))),
        mape=mape,
        r2=float(r2_score(actual, predicted)),
        n=int(actual.size),
    )


def metrics_table(records: List[Metrics]) -> pd.DataFrame:
    """Collect metric records into a comparison table sorted by MAE."""
    frame = pd.DataFrame([record.to_dict() for record in records])
    if frame.empty:
        return frame
    return (frame.drop_duplicates(subset="model", keep="last")
            .set_index("model").sort_values("mae"))


def error_by_level(y_true: Iterable[float], y_pred: Iterable[float]) -> pd.DataFrame:
    """Break absolute error down by consumption band.

    Aggregate metrics hide where a forecaster actually fails. For this series the errors
    concentrate almost entirely in the high-consumption spikes, which this makes explicit.
    """
    actual = np.asarray(y_true, dtype=float)
    residual = np.abs(actual - np.asarray(y_pred, dtype=float))
    bands = pd.cut(actual, [0, 50, 100, 200, 400, np.inf],
                   labels=["<50", "50-100", "100-200", "200-400", ">400"])
    return (pd.Series(residual).groupby(bands, observed=False)
            .agg(mean_abs_error="mean", rows="count").round(2))

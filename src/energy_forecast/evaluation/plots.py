"""Figures and the metrics report.

Plotting is separated from modelling so that figures can be regenerated from saved predictions
without retraining anything, and so the notebook can call exactly the code the batch run uses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import matplotlib
import numpy as np
import pandas as pd

from energy_forecast.utils.logging import get_logger
from energy_forecast.evaluation.metrics import Metrics, error_by_level, metrics_table

logger = get_logger(__name__)


def use_headless_backend() -> None:
    """Switch matplotlib to Agg, for batch runs with no display."""
    matplotlib.use("Agg")


def _plt():  # noqa: ANN202
    """Import pyplot late, so the backend choice above takes effect."""
    import matplotlib.pyplot as plt

    return plt


def save_figure(fig, path: Path) -> Path:
    """Write a figure to disk and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    _plt().close(fig)
    logger.info("figure written to %s", path)
    return path


def plot_actual_vs_predicted(index: pd.DatetimeIndex, actual: np.ndarray,
                             predictions: Mapping[str, np.ndarray],
                             title: str = "Actual vs predicted") -> "matplotlib.figure.Figure":
    """Overlay predictions on the observed series."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(index, actual, lw=0.8, color="black", label="actual")
    for name, values in predictions.items():
        ax.plot(index, values[-len(actual):], lw=0.7, alpha=0.75, label=name)
    ax.set_ylabel("Wh")
    ax.set_title(title)
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_residual_diagnostics(actual: np.ndarray, predicted: np.ndarray,
                              model_name: str) -> "matplotlib.figure.Figure":
    """Four-panel residual analysis.

    Read left to right: does the model track the level; does error grow with the prediction;
    is the error distribution biased or heavy-tailed; and - the strictest test - is there
    autocorrelation left in the residuals, which would mean predictable structure the features
    have not captured.
    """
    plt = _plt()
    residual = np.asarray(actual) - np.asarray(predicted)
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.4))

    ax[0].scatter(predicted, actual, s=3, alpha=0.25)
    limit = [0.0, float(max(np.max(actual), np.max(predicted)))]
    ax[0].plot(limit, limit, "r--", lw=1)
    ax[0].set_xlabel("predicted (Wh)")
    ax[0].set_ylabel("actual (Wh)")
    ax[0].set_title("Predicted vs actual")

    ax[1].scatter(predicted, residual, s=3, alpha=0.25)
    ax[1].axhline(0, color="r", ls="--", lw=1)
    ax[1].set_xlabel("predicted (Wh)")
    ax[1].set_ylabel("residual (Wh)")
    ax[1].set_title("Residuals vs fitted")

    ax[2].hist(residual, bins=70, color="tab:blue")
    ax[2].set_xlabel("residual (Wh)")
    ax[2].set_title("Residual distribution")

    series = pd.Series(residual)
    ax[3].stem([series.autocorr(lag) for lag in range(1, 37)], markerfmt=" ", basefmt=" ")
    ax[3].axhline(0, color="k", lw=0.8)
    ax[3].set_xlabel("lag (steps)")
    ax[3].set_title("Residual ACF")

    for axis in ax:
        axis.grid(alpha=0.3)
    fig.suptitle(f"Residual diagnostics - {model_name}", y=1.04)
    fig.tight_layout()
    return fig


def plot_metric_comparison(table: pd.DataFrame) -> "matplotlib.figure.Figure":
    """Bar chart of each metric across models."""
    plt = _plt()
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    for axis, metric in zip(ax, ["mae", "rmse", "mape", "r2"]):
        axis.barh(table.index, table[metric].values, color="tab:blue")
        axis.set_title(metric.upper())
        axis.tick_params(labelsize=8)
        axis.invert_yaxis()
        axis.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_training_history(histories: Mapping[str, Any]) -> "matplotlib.figure.Figure":
    """Training and validation loss curves, one panel per model."""
    plt = _plt()
    fig, ax = plt.subplots(1, max(len(histories), 1), figsize=(5 * len(histories), 3.6),
                           squeeze=False)
    for axis, (name, history) in zip(ax[0], histories.items()):
        axis.plot(history.history["loss"], label="train")
        axis.plot(history.history["val_loss"], label="validation")
        axis.set_title(f"{name} - MSE loss")
        axis.set_xlabel("epoch")
        axis.legend()
        axis.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def write_report(records: List[Metrics], actual: np.ndarray,
                 predictions: Mapping[str, np.ndarray], path: Path,
                 extra: Optional[Dict[str, object]] = None) -> Path:
    """Write the metrics report as JSON.

    A machine-readable report is what lets a scheduled retrain be compared against the model
    currently in production, rather than eyeballed from a notebook.
    """
    table = metrics_table(records)
    best = table.index[0]
    baseline = table.loc["Persistence", "mae"] if "Persistence" in table.index else None

    payload: Dict[str, object] = {
        "models": [record.to_dict() for record in records],
        "best_model": best,
        "best_mae_wh": float(table.loc[best, "mae"]),
        "test_mean_wh": float(np.mean(actual)),
    }
    if baseline is not None:
        payload["persistence_mae_wh"] = float(baseline)
        payload["improvement_over_persistence_pct"] = float(
            (baseline - table.loc[best, "mae"]) / baseline * 100)
    if best in predictions:
        payload["error_by_level"] = error_by_level(
            actual, predictions[best][-len(actual):]).to_dict()
    if extra:
        payload.update(extra)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    logger.info("metrics report written to %s", path)
    return path


def summarise(records: Iterable[Metrics], actual: np.ndarray) -> str:
    """Render a short human-readable summary of the run."""
    table = metrics_table(list(records))
    best = table.index[0]
    lines = [f"Best model on the test set: {best}",
             f"  MAE  {table.loc[best, 'mae']:.2f} Wh",
             f"  RMSE {table.loc[best, 'rmse']:.2f} Wh",
             f"  MAPE {table.loc[best, 'mape']:.2f} %",
             f"  R2   {table.loc[best, 'r2']:.3f}"]
    if "Persistence" in table.index:
        baseline = table.loc["Persistence", "mae"]
        change = (baseline - table.loc[best, "mae"]) / baseline * 100
        lines.append(f"Against persistence: {change:+.1f}% MAE "
                     f"({baseline:.2f} -> {table.loc[best, 'mae']:.2f} Wh)")
    lines.append(f"Test mean consumption: {np.mean(actual):.1f} Wh")
    return "\n".join(lines)

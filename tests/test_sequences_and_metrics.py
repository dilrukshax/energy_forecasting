"""Windowing alignment and metric correctness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_forecast.evaluation.metrics import compute_metrics, error_by_level, metrics_table
from energy_forecast.training.sequences import make_sequences


def test_window_includes_target_timestamp_safe_features():
    """X[t] is safe because FeatureBuilder has already lagged observations."""
    rows, channels, lookback = 50, 3, 6
    features = np.arange(rows * channels, dtype=float).reshape(rows, channels)
    targets = np.arange(rows, dtype=float)

    windows, aligned = make_sequences(features, targets, lookback)

    assert windows.shape == (rows - lookback + 1, lookback, channels)
    assert aligned.shape == (rows - lookback + 1,)

    for i in (0, 7, rows - lookback):
        np.testing.assert_array_equal(windows[i], features[i:i + lookback])
        assert aligned[i] == targets[i + lookback - 1]


def test_make_sequences_validates_its_inputs():
    """Mismatched lengths and impossible windows must fail loudly."""
    features = np.zeros((10, 2))
    with pytest.raises(ValueError, match="same number of rows"):
        make_sequences(features, np.zeros(9), 3)
    with pytest.raises(ValueError, match="at least"):
        make_sequences(features, np.zeros(10), 11)


def test_metrics_match_hand_computed_values():
    """A metric function that is itself wrong invalidates every conclusion."""
    actual = np.array([100.0, 200.0, 300.0])
    predicted = np.array([110.0, 190.0, 330.0])

    result = compute_metrics(actual, predicted, "test")

    assert result.mae == pytest.approx((10 + 10 + 30) / 3)
    assert result.rmse == pytest.approx(np.sqrt((100 + 100 + 900) / 3))
    assert result.mape == pytest.approx((10 / 100 + 10 / 200 + 30 / 300) / 3 * 100)
    assert result.n == 3


def test_perfect_prediction_scores_perfectly():
    """Sanity anchor for the metric implementation."""
    actual = np.array([10.0, 20.0, 30.0])
    result = compute_metrics(actual, actual, "perfect")
    assert result.mae == 0.0
    assert result.rmse == 0.0
    assert result.r2 == pytest.approx(1.0)


def test_mape_ignores_zero_actuals():
    """Dividing by a zero actual would produce inf and poison the table."""
    actual = np.array([0.0, 100.0])
    predicted = np.array([5.0, 110.0])
    assert np.isfinite(compute_metrics(actual, predicted, "with-zero").mape)


def test_metrics_shape_mismatch_raises():
    """Silently truncating a misaligned prediction would hide a real bug."""
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_metrics(np.zeros(5), np.zeros(4), "bad")


def test_metrics_table_is_sorted_by_mae():
    """The comparison table must put the best model first."""
    records = [compute_metrics(np.array([10.0, 20.0]), np.array([15.0, 25.0]), "worse"),
               compute_metrics(np.array([10.0, 20.0]), np.array([11.0, 21.0]), "better")]
    table = metrics_table(records)
    assert list(table.index) == ["better", "worse"]


def test_error_by_level_covers_every_row():
    """The breakdown must account for all scored rows."""
    actual = np.array([20.0, 80.0, 150.0, 300.0, 600.0])
    predicted = actual + 5
    breakdown = error_by_level(actual, predicted)
    assert breakdown["rows"].sum() == len(actual)
    assert isinstance(breakdown, pd.DataFrame)

"""Data loading, grid enforcement and validation."""

from __future__ import annotations

import pandas as pd
import pytest

from energy_forecast.data import (
    build_dataset,
    enforce_regular_grid,
    load_raw,
    longest_constant_run,
    validate,
)
from energy_forecast.exceptions import DataValidationError


def test_enforce_regular_grid_fills_a_hole(synthetic_frame, synthetic_config):
    """A missing timestamp is inserted, but its target is never invented."""
    punctured = synthetic_frame.drop(synthetic_frame.index[100:103])
    assert len(punctured) == len(synthetic_frame) - 3

    repaired, inserted, filled = enforce_regular_grid(punctured, synthetic_config)

    assert inserted == 3
    assert filled > 0
    assert len(repaired) == len(synthetic_frame)
    assert repaired.index.equals(synthetic_frame.index)
    assert repaired.loc[synthetic_frame.index[100:103], synthetic_config.target].isna().all()
    assert not repaired.drop(columns=synthetic_config.target).isna().any().any()


def test_long_gaps_are_not_fabricated(synthetic_frame, synthetic_config):
    """A dropout longer than the cap stays NaN rather than being invented."""
    limit = int(synthetic_config.data["max_forward_fill_steps"])
    punctured = synthetic_frame.drop(synthetic_frame.index[200:200 + limit * 3])

    repaired, _, _ = enforce_regular_grid(punctured, synthetic_config)

    assert repaired.isna().any().any(), "an over-long gap must not be fully filled"


def test_validate_rejects_duplicate_timestamps(synthetic_frame, synthetic_config):
    """Duplicate timestamps break every lag feature, so they must be fatal."""
    duplicated = pd.concat([synthetic_frame, synthetic_frame.iloc[[5]]]).sort_index()
    with pytest.raises(DataValidationError, match="duplicate"):
        validate(duplicated, synthetic_config, strict=True)


def test_validate_rejects_irregular_spacing(synthetic_frame, synthetic_config):
    """An irregular interval invalidates the fixed-step assumption."""
    irregular = synthetic_frame.drop(synthetic_frame.index[50])
    with pytest.raises(DataValidationError, match="irregular"):
        validate(irregular, synthetic_config, strict=True)


def test_longest_constant_run_detects_a_frozen_sensor():
    """A stuck sensor reports plausible values, so nulls will not reveal it."""
    series = pd.Series([1.0, 2.0] + [5.0] * 12 + [3.0])
    assert longest_constant_run(series) == 12


@pytest.mark.slow
def test_real_dataset_matches_expected_shape(config, raw_csv_path):
    """Guard against a silent schema change in the source file."""
    frame = load_raw(config, raw_csv_path)
    assert config.target in frame.columns
    assert "rv1" not in frame.columns and "rv2" not in frame.columns
    assert isinstance(frame.index, pd.DatetimeIndex)


@pytest.mark.slow
def test_real_dataset_is_a_clean_grid(config, raw_csv_path):
    """The supplied file should need no repair; if that changes we want to know."""
    _, report = build_dataset(config, raw_csv_path)
    assert report.duplicate_timestamps == 0
    assert report.missing_timestamps == 0
    assert report.missing_cells == 0

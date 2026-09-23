"""Data loading, schema validation and time-grid enforcement.

The raw CSV is treated as immutable. Everything here is a pure transformation from it, so the
processed dataset can always be regenerated and never has to be trusted as an input.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from energy_forecast.config import Config
from energy_forecast.exceptions import DataValidationError
from energy_forecast.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DataQualityReport:
    """Summary of the checks run against a freshly loaded dataset.

    Persisted alongside the model so that a later run can be compared against the data the
    model was trained on - a silent schema or frequency change is a common production failure.
    """

    n_rows: int
    start: pd.Timestamp
    end: pd.Timestamp
    duplicate_timestamps: int
    missing_timestamps: int
    missing_cells: int
    filled_sensor_cells: int
    longest_constant_runs: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""
        return {
            "n_rows": self.n_rows,
            "start": str(self.start),
            "end": str(self.end),
            "duplicate_timestamps": self.duplicate_timestamps,
            "missing_timestamps": self.missing_timestamps,
            "missing_cells": self.missing_cells,
            "filled_sensor_cells": self.filled_sensor_cells,
            "longest_constant_runs": self.longest_constant_runs,
        }


def load_raw(config: Config, path: Path | None = None) -> pd.DataFrame:
    """Read the raw CSV into a time-indexed frame.

    Args:
        config: Loaded configuration.
        path: Optional override for the raw data location.

    Returns:
        A frame indexed by timestamp and sorted ascending, with the configured noise columns
        removed.

    Raises:
        DataValidationError: if the target or timestamp column is absent.

    """
    source = path or config.raw_path
    timestamp_col = config.data["timestamp_column"]

    frame = pd.read_csv(source, parse_dates=[timestamp_col])
    logger.info("loaded %s rows from %s", len(frame), source)

    for required in (timestamp_col, config.target):
        if required not in frame.columns:
            raise DataValidationError(f"required column {required!r} missing from {source}")

    frame = frame.sort_values(timestamp_col).set_index(timestamp_col)

    drop = {col: why for col, why in (config.data.get("drop_columns") or {}).items()
            if col in frame.columns}
    if drop:
        for col, why in drop.items():
            logger.info("dropping column %s (%s)", col, why)
        frame = frame.drop(columns=list(drop))

    return frame


def longest_constant_run(series: pd.Series) -> int:
    """Length of the longest run of identical consecutive values.

    A frozen sensor reports a plausible value repeatedly, so it is invisible to a null check.
    This is the cheapest way to surface one.
    """
    changed = series.ne(series.shift()).cumsum()
    return int(series.groupby(changed).transform("size").max())


def enforce_regular_grid(frame: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, int, int]:
    """Reindex onto a complete time grid and forward-fill short sensor gaps.

    Every lag and rolling feature assumes a fixed step size. A timestamp that is simply absent
    from the file is invisible to ``isna()`` but silently corrupts those features, so the grid is
    enforced explicitly rather than assumed.

    Sensor forward fill is capped at ``data.max_forward_fill_steps`` consecutive steps. The
    target is never filled, because a missing observation is not a training label.

    Args:
        frame: Time-indexed frame.
        config: Loaded configuration.

    Returns:
        Tuple of (regularised frame, timestamps inserted, sensor cells forward-filled).

    """
    grid = pd.date_range(frame.index.min(), frame.index.max(), freq=config.frequency)
    inserted = len(grid) - len(frame.index.intersection(grid))

    regular = frame.reindex(grid)
    regular.index.name = frame.index.name

    before = int(regular.isna().sum().sum())
    limit = int(config.data.get("max_forward_fill_steps", 6))
    # Never invent target labels. Missing sensor readings can use the most recent observation,
    # but a missing appliance reading must stay missing and be excluded from training.
    sensor_columns = regular.columns.drop(config.target)
    regular.loc[:, sensor_columns] = regular.loc[:, sensor_columns].ffill(limit=limit)
    after = int(regular.isna().sum().sum())

    logger.info("grid enforcement: %s timestamps inserted, %s sensor cells forward-filled",
                inserted, before - after)
    return regular, inserted, before - after


def validate(frame: pd.DataFrame, config: Config, strict: bool = True) -> None:
    """Assert the invariants the rest of the pipeline relies on.

    Args:
        frame: Regularised frame.
        config: Loaded configuration.
        strict: When true, a violation raises; otherwise it is logged as a warning.

    Raises:
        DataValidationError: in strict mode, on duplicate timestamps, a non-monotonic index,
            an irregular step size or a target that is entirely missing.

    """
    problems: list[str] = []

    if frame.index.duplicated().any():
        problems.append(f"{int(frame.index.duplicated().sum())} duplicate timestamps")
    if not frame.index.is_monotonic_increasing:
        problems.append("index is not sorted ascending")

    steps = frame.index.to_series().diff().dropna().unique()
    expected = pd.Timedelta(pd.tseries.frequencies.to_offset(config.frequency))
    if len(steps) > 1 or (len(steps) == 1 and steps[0] != expected):
        problems.append(f"irregular sampling interval: {steps[:5]}")

    if frame[config.target].isna().all():
        problems.append("target column is entirely missing")

    if problems:
        message = "; ".join(problems)
        if strict:
            raise DataValidationError(message)
        logger.warning("data validation: %s", message)


def build_dataset(config: Config, path: Path | None = None
                  ) -> tuple[pd.DataFrame, DataQualityReport]:
    """Run the full load -> regularise -> validate sequence.

    Returns:
        Tuple of (analysis-ready frame, quality report).

    """
    raw = load_raw(config, path)
    duplicates = int(raw.index.duplicated().sum())
    if duplicates:
        raise DataValidationError(f"{duplicates} duplicate timestamps in raw data")
    grid = pd.date_range(raw.index.min(), raw.index.max(), freq=config.frequency)
    missing_timestamps = len(grid.difference(raw.index))
    missing_cells = int(raw.isna().sum().sum())

    frame, _, filled = enforce_regular_grid(raw, config)
    validate(frame, config, strict=True)

    runs = {col: longest_constant_run(frame[col]) for col in frame.columns}
    top_runs = dict(sorted(runs.items(), key=lambda kv: kv[1], reverse=True)[:10])

    report = DataQualityReport(
        n_rows=len(frame),
        start=frame.index.min(),
        end=frame.index.max(),
        duplicate_timestamps=duplicates,
        missing_timestamps=missing_timestamps,
        missing_cells=missing_cells,
        filled_sensor_cells=filled,
        longest_constant_runs=top_runs,
    )
    logger.info("dataset ready: %s rows, %s -> %s", report.n_rows, report.start, report.end)
    return frame, report

"""Chronological partitioning.

A random split is invalid for this problem. Adjacent 10-minute rows correlate at roughly 0.75,
so shuffling puts a row's immediate neighbour in the training set while the row itself sits in
the test set. The model then interpolates instead of forecasting and the metrics mean nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from energy_forecast.config import Config
from energy_forecast.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Split:
    """A chronological train/validation/test partition.

    Attributes:
        X_train, y_train: Earliest block, used to fit models and every preprocessing artefact.
        X_val, y_val: Middle block, used for early stopping and hyper-parameter selection.
        X_test, y_test: Final block, touched once, to report.
        val_start, test_start: Integer positions of the two boundaries, needed to align the
            windowed sequences with the flat rows.

    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    val_start: int
    test_start: int

    @property
    def summary(self) -> pd.DataFrame:
        """Row counts and date ranges for each partition."""
        rows = []
        for name, X in (("train", self.X_train), ("validation", self.X_val),
                        ("test", self.X_test)):
            rows.append({"split": name, "rows": len(X),
                         "start": X.index.min(), "end": X.index.max()})
        return pd.DataFrame(rows).set_index("split")


def chronological_split(X: pd.DataFrame, y: pd.Series, config: Config) -> Split:
    """Partition by time, never by shuffling.

    The test block is the final ``split.test_fraction`` of the timeline. The validation block is
    the final ``split.validation_fraction`` of what remains, so the ordering is always
    train < validation < test in time.

    Args:
        X: Design matrix, sorted ascending by timestamp.
        y: Aligned target.
        config: Loaded configuration.

    Returns:
        A :class:`Split`.

    Raises:
        ValueError: if X and y are misaligned or the index is not sorted.

    """
    if not X.index.equals(y.index):
        raise ValueError("X and y must share an identical index")
    if not X.index.is_monotonic_increasing:
        raise ValueError("index must be sorted ascending before splitting")

    n = len(X)
    test_start = int(n * (1 - float(config.split["test_fraction"])))
    val_start = int(test_start * (1 - float(config.split["validation_fraction"])))

    split = Split(
        X_train=X.iloc[:val_start], y_train=y.iloc[:val_start],
        X_val=X.iloc[val_start:test_start], y_val=y.iloc[val_start:test_start],
        X_test=X.iloc[test_start:], y_test=y.iloc[test_start:],
        val_start=val_start, test_start=test_start,
    )
    logger.info("chronological split -> train %s, val %s, test %s",
                len(split.X_train), len(split.X_val), len(split.X_test))
    return split

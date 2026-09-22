"""Windowing for the recurrent models.

A recurrent network consumes a window of consecutive rows, not a flat row. The window ending at
``t - horizon`` predicts the target at ``t``.

Windows are cut once over the whole timeline and then partitioned by *target position*, reusing
the boundaries computed in :mod:`splitting`. A test window may therefore reach back into
training-period rows. That is correct rather than leakage: at prediction time the recent past
genuinely is available. The reverse - a window containing rows at or after its own target -
cannot occur, and :func:`make_sequences` is covered by a test that asserts it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from energy_forecast.utils.logging import get_logger
from energy_forecast.data.splitting import Split

logger = get_logger(__name__)


@dataclass(frozen=True)
class SequenceData:
    """Windowed tensors for the three partitions.

    Attributes:
        X_train, X_val, X_test: Arrays of shape (windows, lookback, features).
        y_train, y_val, y_test: Targets in model space.
        test_index: Timestamps of the test targets, for plotting.
        lookback: Window length in steps.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_index: pd.DatetimeIndex
    lookback: int

    @property
    def n_features(self) -> int:
        """Number of channels per timestep."""
        return int(self.X_train.shape[2])

    @property
    def input_shape(self) -> tuple[int, int]:
        """Keras input shape for one window."""
        return self.lookback, self.n_features


def make_sequences(features: np.ndarray, targets: np.ndarray,
                   lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Turn a 2-D feature matrix into overlapping 3-D windows.

    Window ``i`` spans rows ``i .. i + lookback - 1`` and is paired with the target at row
    ``i + lookback``: every window ends strictly before the value it predicts.

    Args:
        features: Array of shape (rows, channels).
        targets: Array of shape (rows,).
        lookback: Window length in steps.

    Returns:
        Tuple of (windows, aligned targets).

    Raises:
        ValueError: if the inputs disagree on length or are shorter than the window.
    """
    if features.shape[0] != targets.shape[0]:
        raise ValueError("features and targets must have the same number of rows")
    if features.shape[0] <= lookback:
        raise ValueError(f"need more than {lookback} rows to build a single window")

    windows = np.lib.stride_tricks.sliding_window_view(
        features, (lookback, features.shape[1])).squeeze(1)[:-1]
    return np.ascontiguousarray(windows), np.asarray(targets)[lookback:]


def build_sequence_data(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                        y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                        split: Split, index: pd.DatetimeIndex,
                        lookback: int) -> SequenceData:
    """Window the whole timeline once, then partition by target position.

    Args:
        X_train, X_val, X_test: Scaled, selected feature matrices for the three blocks.
        y_train, y_val, y_test: Targets in model space for the three blocks.
        split: The chronological split, for its boundary positions.
        index: Timestamp index of the full design matrix.
        lookback: Window length in steps.

    Returns:
        A :class:`SequenceData`.
    """
    features = np.vstack([X_train, X_val, X_test])
    targets = np.concatenate([y_train, y_val, y_test])

    windows, aligned = make_sequences(features, targets, lookback)
    target_positions = np.arange(lookback, len(targets))

    is_train = target_positions < split.val_start
    is_val = (target_positions >= split.val_start) & (target_positions < split.test_start)
    is_test = target_positions >= split.test_start

    data = SequenceData(
        X_train=windows[is_train], y_train=aligned[is_train],
        X_val=windows[is_val], y_val=aligned[is_val],
        X_test=windows[is_test], y_test=aligned[is_test],
        test_index=index[target_positions[is_test]],
        lookback=lookback,
    )
    logger.info("sequences: train %s, val %s, test %s (lookback %s)",
                data.X_train.shape, data.X_val.shape, data.X_test.shape, lookback)
    return data

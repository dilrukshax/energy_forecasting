"""Winsorising and scaling, fitted on training rows only.

Fitting any statistic on data the model will later be scored against is leakage, even when the
statistic looks innocuous: a scaler that has seen the test period's range has been told
something about the future. Everything here is fitted once, on the training block, and then
applied unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from energy_forecast.config import Config
from energy_forecast.exceptions import NotFittedError
from energy_forecast.utils.logging import get_logger

logger = get_logger(__name__)

# Categorical, cyclical and indicator features are left alone: clipping an indicator or a
# sine wave to an IQR fence is meaningless.
NEVER_CLIP = frozenset({
    "hour", "day_of_week", "month", "is_weekend", "nsm", "hour_sin", "hour_cos",
    "dow_sin", "dow_cos", "is_holiday", "is_non_working", "evening_peak_flag",
    "hour_x_weekend",
})


def iqr_bounds(series: pd.Series, multiplier: float = 1.5) -> tuple[float, float]:
    """Tukey fences at ``multiplier`` times the interquartile range."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - multiplier * iqr), float(q3 + multiplier * iqr)


@dataclass
class Winsoriser:
    """Clips predictors to fences learned from the training split.

    Target spikes are genuine consumption events and are never clipped - see
    :class:`TargetTransformer`, which handles the target's skew with a log transform instead.
    An extreme *predictor* value is more likely to be an artefact, and one wild reading can
    distort a scaler that later feeds a neural network.
    """

    multiplier: float = 3.0
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)

    def fit(self, X: pd.DataFrame) -> Winsoriser:
        """Learn fences from the training rows."""
        self.columns = [
            c for c in X.columns
            if c not in NEVER_CLIP and not c.startswith("app_")
        ]
        self.bounds = {c: iqr_bounds(X[c], self.multiplier) for c in self.columns}
        capped = float(np.mean([
            ((X[c] < lo) | (X[c] > hi)).mean() for c, (lo, hi) in self.bounds.items()
        ])) if self.columns else 0.0
        logger.info("winsoriser fitted on %s columns; mean %.2f%% of training values capped",
                    len(self.columns), capped * 100)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned fences."""
        if not self.bounds:
            raise NotFittedError("Winsoriser must be fitted before transform")
        out = X.copy()
        for column, (low, high) in self.bounds.items():
            if column in out.columns:
                out[column] = out[column].clip(low, high)
        return out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit then transform, for the training block only."""
        return self.fit(X).transform(X)


@dataclass
class TargetTransformer:
    """log1p + standardisation for the target, with an exact inverse.

    The target's skew is about 3.4: a raw-scale squared-error loss is dominated by a handful of
    evening spikes. ``log1p`` compresses them without discarding any observation, and the
    standardisation afterwards keeps the loss surface well conditioned for gradient descent.

    All metrics are computed after :meth:`inverse`, so reported errors are in Wh.
    """

    log_transform: bool = True
    scaler: StandardScaler = field(default_factory=StandardScaler)

    def fit(self, y: Iterable[float]) -> TargetTransformer:
        """Fit on training targets only."""
        values = self._to_log(np.asarray(y, dtype=float))
        self.scaler.fit(values.reshape(-1, 1))
        return self

    def transform(self, y: Iterable[float]) -> np.ndarray:
        """Map Wh into model space."""
        values = self._to_log(np.asarray(y, dtype=float))
        return self.scaler.transform(values.reshape(-1, 1)).ravel()

    def inverse(self, y: Iterable[float]) -> np.ndarray:
        """Map model space back to Wh."""
        values = self.scaler.inverse_transform(np.asarray(y, dtype=float).reshape(-1, 1)).ravel()
        return np.expm1(values) if self.log_transform else values

    def fit_transform(self, y: Iterable[float]) -> np.ndarray:
        """Fit then transform."""
        return self.fit(y).transform(y)

    def _to_log(self, values: np.ndarray) -> np.ndarray:
        return np.log1p(values) if self.log_transform else values


@dataclass
class Preprocessor:
    """Bundles the winsoriser, the feature scaler and the target transformer.

    Saved as a single artefact so that inference loads exactly the transformations training
    used. A model shipped without its preprocessing is not reproducible.
    """

    config: Config
    winsoriser: Winsoriser = field(init=False)
    feature_scaler: StandardScaler = field(init=False)
    target_transformer: TargetTransformer = field(init=False)
    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Create the unfitted transformations from configuration."""
        settings = self.config.preprocessing
        self.winsoriser = Winsoriser(float(settings.get("winsorise_iqr_multiplier", 3.0)))
        self.feature_scaler = StandardScaler()
        self.target_transformer = TargetTransformer(
            bool(settings.get("target_log_transform", True)))

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> Preprocessor:
        """Fit every artefact on the training block."""
        self.feature_names = list(X_train.columns)
        self.feature_scaler.fit(self.winsoriser.fit_transform(X_train))
        self.target_transformer.fit(y_train)
        logger.info("preprocessor fitted on %s training rows", len(X_train))
        return self

    def transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """Winsorise then standardise, in the column order seen at fit time."""
        if list(X.columns) != self.feature_names:
            raise ValueError("feature columns differ from those seen during fit")
        return self.feature_scaler.transform(self.winsoriser.transform(X))

    def save(self, path: Path) -> Path:
        """Persist the fitted artefacts next to the model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("preprocessor saved to %s", path)
        return path

    @staticmethod
    def load(path: Path) -> Preprocessor:
        """Restore a persisted preprocessor."""
        return joblib.load(path)

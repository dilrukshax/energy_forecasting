"""Leakage-safe feature engineering.

Temporal leakage is the failure mode this module exists to prevent: a feature that could not
actually be computed at prediction time makes validation scores look excellent and production
performance collapse. The usual vector is a rolling window that quietly includes the current
observation.

Two rules are enforced mechanically rather than by convention:

1. Every feature derived from *observed* data is shifted by at least ``horizon`` steps, so it
   depends only on rows strictly before the timestamp it describes.
2. Only features flagged :data:`Availability.KNOWN_IN_ADVANCE` - the calendar, which is knowable
   for any future timestamp - may use the current row.

:func:`feature_audit` reports which rule each feature falls under, and
``tests/test_features_leakage.py`` verifies rule 1 empirically by corrupting the future and
asserting the past does not move.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from energy_forecast.config import Config
from energy_forecast.exceptions import LeakageError
from energy_forecast.utils.logging import get_logger

logger = get_logger(__name__)

INDOOR_TEMPERATURE = ["T1", "T2", "T3", "T4", "T5", "T7", "T8", "T9"]
INDOOR_HUMIDITY = ["RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_7", "RH_8", "RH_9"]


class Availability(str, Enum):
    """Whether a feature can be computed at serving time, and from what."""

    KNOWN_IN_ADVANCE = "known_in_advance"
    LAGGED_OBSERVATION = "lagged_observation"


class Family(str, Enum):
    """Grouping used for reporting and for the feature audit."""

    EXOGENOUS = "exogenous"
    TARGET_LAG = "target_lag"
    ROLLING = "rolling"
    DIFFERENCE = "difference"
    CALENDAR = "calendar"
    DOMAIN = "domain"
    INTERACTION = "interaction"


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata describing one engineered feature.

    Attributes:
        name: Column name in the design matrix.
        family: Which group the feature belongs to.
        availability: Whether it needs observed history or is knowable in advance.
        min_lag: Smallest number of steps between the feature's newest input and the row it
            describes. Must be >= the horizon for any observed quantity.
        description: Why the feature exists, for the model card.
    """

    name: str
    family: Family
    availability: Availability
    min_lag: int
    description: str


class FeatureBuilder:
    """Builds the design matrix and records the provenance of every column.

    Args:
        config: Loaded configuration. Lags, rolling windows and holidays all come from it.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.target = config.target
        self.horizon = int(config.features.get("horizon", 1))
        self.target_lags: List[int] = list(config.features["target_lags"])
        self.rolling_windows: Dict[str, int] = dict(config.features["rolling_windows"])
        self.holidays = pd.to_datetime(list(config.features.get("holidays", [])))
        self.specs: List[FeatureSpec] = []

    # -- internals ------------------------------------------------------------------------
    def _record(self, name: str, family: Family, availability: Availability,
                min_lag: int, description: str) -> None:
        """Register a feature's provenance, rejecting any that breaks the leakage rule."""
        if availability is Availability.LAGGED_OBSERVATION and min_lag < self.horizon:
            raise LeakageError(
                f"feature {name!r} claims min_lag={min_lag} but the horizon is {self.horizon}; "
                "an observed quantity may not use data from the row it predicts"
            )
        self.specs.append(FeatureSpec(name, family, availability, min_lag, description))

    def _past(self, series: pd.Series) -> pd.Series:
        """Shift an observed series by the horizon, so it is available at prediction time."""
        return series.shift(self.horizon)

    # -- public API -----------------------------------------------------------------------
    def build(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Construct the design matrix and the aligned target.

        Args:
            frame: Regularised, time-indexed frame containing the target and the sensors.

        Returns:
            Tuple of (X, y) sharing an index, with warm-up rows containing NaN dropped.
        """
        self.specs = []
        features = pd.DataFrame(index=frame.index)
        target_history = self._past(frame[self.target])
        exogenous = [c for c in frame.columns if c != self.target]

        self._add_exogenous(features, frame, exogenous)
        self._add_target_lags(features, frame)
        self._add_rolling(features, frame, target_history)
        self._add_differences(features, frame)
        self._add_calendar(features, frame)
        self._add_domain(features, frame)
        self._add_interactions(features, frame)

        combined = features.join(frame[self.target].rename("__target__")).dropna()
        X = combined.drop(columns="__target__")
        y = combined["__target__"].rename(self.target)

        logger.info("design matrix: %s rows x %s features (%s warm-up rows dropped)",
                    len(X), X.shape[1], len(frame) - len(X))
        return X, y

    # -- feature families -------------------------------------------------------------------
    def _add_exogenous(self, out: pd.DataFrame, frame: pd.DataFrame, columns: List[str]) -> None:
        """Sensor readings, lagged by the horizon.

        The sensors are lagged rather than used at time t because at prediction time the current
        reading has not been collected yet. Treating them as contemporaneous would be a
        serving-time impossibility disguised as a feature.
        """
        for column in columns:
            name = f"{column}_lag{self.horizon}"
            out[name] = self._past(frame[column])
            self._record(name, Family.EXOGENOUS, Availability.LAGGED_OBSERVATION, self.horizon,
                         f"{column} observed {self.horizon} step(s) earlier")

    def _add_target_lags(self, out: pd.DataFrame, frame: pd.DataFrame) -> None:
        """Past values of the target at the lags chosen from the autocorrelation analysis."""
        for lag in self.target_lags:
            if lag < self.horizon:
                continue
            name = f"app_lag{lag}"
            out[name] = frame[self.target].shift(lag)
            self._record(name, Family.TARGET_LAG, Availability.LAGGED_OBSERVATION, lag,
                         f"target observed {lag} step(s) earlier")

    def _add_rolling(self, out: pd.DataFrame, frame: pd.DataFrame,
                     target_history: pd.Series) -> None:
        """Rolling statistics over strictly past target values.

        ``target_history`` is already shifted, so a window of width w ends at t-horizon and can
        never include the value being predicted. This is the single most common place temporal
        leakage is introduced.
        """
        for label, width in self.rolling_windows.items():
            window = target_history.rolling(int(width))
            for stat, func in (("mean", window.mean), ("std", window.std),
                               ("max", window.max), ("min", window.min)):
                name = f"app_{stat}_{label}"
                out[name] = func()
                self._record(name, Family.ROLLING, Availability.LAGGED_OBSERVATION, self.horizon,
                             f"rolling {stat} of the target over the previous {label}")

        for column, label, width in (("lights", "1h", 6), ("T_out", "3h", 18)):
            if column not in frame.columns:
                continue
            name = f"{column}_mean_{label}"
            out[name] = self._past(frame[column]).rolling(width).mean()
            self._record(name, Family.ROLLING, Availability.LAGGED_OBSERVATION, self.horizon,
                         f"rolling mean of {column} over the previous {label}")

    def _add_differences(self, out: pd.DataFrame, frame: pd.DataFrame) -> None:
        """Momentum features: level alone cannot express direction of travel."""
        h = self.horizon
        out["app_diff_1"] = frame[self.target].shift(h) - frame[self.target].shift(h + 1)
        self._record("app_diff_1", Family.DIFFERENCE, Availability.LAGGED_OBSERVATION, h,
                     "change in the target over the most recent completed step")

        out["app_diff_6"] = frame[self.target].shift(h) - frame[self.target].shift(h + 6)
        self._record("app_diff_6", Family.DIFFERENCE, Availability.LAGGED_OBSERVATION, h,
                     "change in the target over the previous hour")

        if "app_mean_24h" in out.columns:
            out["app_dev_from_24h"] = frame[self.target].shift(h) - out["app_mean_24h"]
            self._record("app_dev_from_24h", Family.DIFFERENCE,
                         Availability.LAGGED_OBSERVATION, h,
                         "deviation of the last observation from its 24h mean")

    def _add_calendar(self, out: pd.DataFrame, frame: pd.DataFrame) -> None:
        """Clock and calendar features.

        These are the one family allowed to use the current row: the hour and weekday of a
        future timestamp are known without observing anything.
        """
        index = frame.index
        seconds_since_midnight = index.hour * 3600 + index.minute * 60

        plain = {
            "hour": index.hour,
            "day_of_week": index.dayofweek,
            "month": index.month,
            "is_weekend": (index.dayofweek >= 5).astype(int),
            "nsm": seconds_since_midnight,
        }
        for name, values in plain.items():
            out[name] = values
            self._record(name, Family.CALENDAR, Availability.KNOWN_IN_ADVANCE, 0,
                         "calendar attribute of the predicted timestamp")

        cyclical = {
            "hour_sin": np.sin(2 * np.pi * seconds_since_midnight / 86400),
            "hour_cos": np.cos(2 * np.pi * seconds_since_midnight / 86400),
            "dow_sin": np.sin(2 * np.pi * index.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * index.dayofweek / 7),
        }
        for name, values in cyclical.items():
            out[name] = values
            self._record(name, Family.CALENDAR, Availability.KNOWN_IN_ADVANCE, 0,
                         "cyclical encoding so midnight is adjacent to 23:50, not maximally far")

    def _add_domain(self, out: pd.DataFrame, frame: pd.DataFrame) -> None:
        """Domain knowledge: the dwelling is in Belgium, so its public holidays apply."""
        index = frame.index
        out["is_holiday"] = np.isin(index.normalize(), self.holidays).astype(int)
        self._record("is_holiday", Family.DOMAIN, Availability.KNOWN_IN_ADVANCE, 0,
                     "Belgian public holiday; behaves like a weekend regardless of weekday")

        out["is_non_working"] = ((out["is_weekend"] == 1) | (out["is_holiday"] == 1)).astype(int)
        self._record("is_non_working", Family.DOMAIN, Availability.KNOWN_IN_ADVANCE, 0,
                     "weekend or public holiday")

        out["evening_peak_flag"] = ((index.hour >= 17) & (index.hour <= 20)).astype(int)
        self._record("evening_peak_flag", Family.DOMAIN, Availability.KNOWN_IN_ADVANCE, 0,
                     "the 17:00-20:00 window the EDA identified as the daily peak")

    def _add_interactions(self, out: pd.DataFrame, frame: pd.DataFrame) -> None:
        """Products and contrasts that carry more signal than either input alone."""
        h = self.horizon
        available_t = [c for c in INDOOR_TEMPERATURE if c in frame.columns]
        available_rh = [c for c in INDOOR_HUMIDITY if c in frame.columns]

        def register(name: str, values: pd.Series, description: str,
                     availability: Availability = Availability.LAGGED_OBSERVATION) -> None:
            out[name] = values
            self._record(name, Family.INTERACTION, availability,
                         h if availability is Availability.LAGGED_OBSERVATION else 0, description)

        if available_t:
            register("T_indoor_mean", frame[available_t].shift(h).mean(axis=1),
                     "mean indoor temperature across rooms")
            register("T_indoor_range",
                     frame[available_t].shift(h).max(axis=1) - frame[available_t].shift(h).min(axis=1),
                     "spread between the warmest and coolest room")
        if available_rh:
            register("RH_indoor_mean", frame[available_rh].shift(h).mean(axis=1),
                     "mean indoor relative humidity across rooms")
        if available_t and available_rh:
            register("T_x_RH_indoor", out["T_indoor_mean"] * out["RH_indoor_mean"],
                     "indoor comfort proxy: temperature and humidity act jointly, not separately")
        if {"T_out", "RH_out"} <= set(frame.columns):
            register("T_out_x_RH_out", self._past(frame["T_out"]) * self._past(frame["RH_out"]),
                     "outdoor comfort proxy")
        if available_t and "T_out" in frame.columns:
            register("indoor_minus_outdoor_T", out["T_indoor_mean"] - self._past(frame["T_out"]),
                     "thermal gradient driving heating and ventilation demand")
        if {"T_out", "Tdewpoint"} <= set(frame.columns):
            register("dewpoint_spread", self._past(frame["T_out"]) - self._past(frame["Tdewpoint"]),
                     "dew-point depression, a humidity measure independent of temperature")
        if "hour_sin" in out.columns and "is_weekend" in out.columns:
            register("hour_x_weekend", out["hour_sin"] * out["is_weekend"],
                     "the daily shape differs between weekdays and weekends",
                     Availability.KNOWN_IN_ADVANCE)

    # -- reporting -----------------------------------------------------------------------
    def audit(self) -> pd.DataFrame:
        """Return the serving-time availability audit for the features just built.

        One row per feature, answering the question the leakage literature recommends asking of
        every input: *does this exist at serving time, and from what?*
        """
        if not self.specs:
            raise RuntimeError("call build() before audit()")
        return pd.DataFrame([{
            "feature": spec.name,
            "family": spec.family.value,
            "availability": spec.availability.value,
            "min_lag_steps": spec.min_lag,
            "description": spec.description,
        } for spec in self.specs]).set_index("feature")


def build_features(frame: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, pd.Series]:
    """Convenience wrapper around :class:`FeatureBuilder`.

    Args:
        frame: Regularised, time-indexed frame.
        config: Loaded configuration.

    Returns:
        Tuple of (X, y).
    """
    return FeatureBuilder(config).build(frame)

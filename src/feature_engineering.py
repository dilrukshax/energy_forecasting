"""Build target-time rows using only measurements strictly before their timestamp."""
import numpy as np
import pandas as pd
from .data_preprocessing import LOCAL, TARGET

BASE_LAGS = [1, 3, 6, 18, 144]
MAX_LAG = 144
MAX_LOOKBACK = 36
WARMUP = MAX_LAG + MAX_LOOKBACK - 1


def choose_lags(training_target):
    """Add two high-ACF short lags to interpretable 10m/30m/1h/3h/1d lags.

    ACF is a training-only heuristic, not proof that these are globally optimal.
    """
    acf = pd.Series({lag: training_target.autocorr(lag=lag) for lag in range(1, 145)}, name="acf")
    candidates = acf.loc[1:36].drop(BASE_LAGS, errors="ignore")
    extra = candidates.abs().nlargest(2).index.tolist()
    return sorted(set(BASE_LAGS + extra)), acf


def make_features(frame, lags):
    """F[t] predicts y[t] from observations <=t-1 plus the known calendar at t.

    Rolling means use shift(1) BEFORE rolling. Sensor forward-fill uses only
    previous readings, with a maximum of six records (one hour).
    """
    features = pd.DataFrame(index=frame.index)
    past = frame[LOCAL].shift(1).ffill(limit=6)
    for name in LOCAL:
        features[f"{name}_lag1"] = past[name]
    for lag in lags:
        features[f"energy_lag_{lag}"] = frame[TARGET].shift(lag)
    historical_energy = frame[TARGET].shift(1)
    for window in (6, 18):
        rolling = historical_energy.rolling(window, min_periods=window)
        features[f"energy_mean_{window}"] = rolling.mean()
        features[f"energy_std_{window}"] = rolling.std(ddof=0)
        features[f"energy_max_{window}"] = rolling.max()
    features["energy_change_10m"] = frame[TARGET].shift(1) - frame[TARGET].shift(2)
    features["energy_change_30m"] = frame[TARGET].shift(1) - frame[TARGET].shift(4)
    minutes = frame.index.hour * 60 + frame.index.minute
    day_sunday_zero = (frame.index.dayofweek + 1) % 7
    features["NSM"] = minutes * 60 + frame.index.second
    features["hour"] = frame.index.hour
    features["Day_of_week"] = day_sunday_zero
    features["month"] = frame.index.month
    features["is_weekend"] = (frame.index.dayofweek >= 5).astype(int)
    features["hour_sin"] = np.sin(2 * np.pi * minutes / 1440)
    features["hour_cos"] = np.cos(2 * np.pi * minutes / 1440)
    features["dow_sin"] = np.sin(2 * np.pi * day_sunday_zero / 7)
    features["dow_cos"] = np.cos(2 * np.pi * day_sunday_zero / 7)
    indoor_ids = [1, 2, 3, 4, 5, 7, 8, 9]  # T6 is OUTDOORS, per UCI.
    features["indoor_temp_mean"] = past[[f"T{i}" for i in indoor_ids]].mean(axis=1)
    features["indoor_rh_mean"] = past[[f"RH_{i}" for i in indoor_ids]].mean(axis=1)
    features["indoor_outdoor_temp_gap"] = features["indoor_temp_mean"] - past["T6"]
    features["indoor_outdoor_rh_gap"] = features["indoor_rh_mean"] - past["RH_6"]
    features["kitchen_temp_x_rh"] = past["T1"] * past["RH_1"] / 100
    features["living_temp_x_rh"] = past["T2"] * past["RH_2"] / 100
    features["lights_x_evening"] = past["lights"] * ((frame.index.hour >= 18) & (frame.index.hour <= 23))
    return features.astype(float)


def select_features(importances, top_k=24):
    """Keep training-ranked features and a small predeclared time/energy core."""
    mandatory = ["energy_lag_1", "energy_lag_3", "energy_mean_6", "hour_sin", "hour_cos", "is_weekend"]
    selected = list(dict.fromkeys(mandatory + importances.sort_values(ascending=False).head(top_k).index.tolist()))
    return selected

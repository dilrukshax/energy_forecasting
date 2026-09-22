"""Feature engineering and selection for next-step appliance energy forecasting."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge

TARGET = "Appliances"
STEPS_PER_DAY = 144
STEPS_PER_WEEK = STEPS_PER_DAY * 7


def build_features(frame, target=TARGET):
    """Build the leakage-safe design matrix.

    All sensor, lag, rolling and difference features are computed from values strictly before the
    timestamp of the row they belong to (shift >= 1). Calendar features use the timestamp itself,
    which is known ahead of time.
    Returns (X, y) aligned on the same index, with warm-up rows dropped.
    """
    X = pd.DataFrame(index=frame.index)
    past = frame[target].shift(1)  # target history, strictly <= t-1
    exog = [c for c in frame.columns if c != target]

    # --- 1. Exogenous sensors, lagged one step ------------------------------------------
    for col in exog:
        X[f"{col}_lag1"] = frame[col].shift(1)

    # --- 2. Lagged target ---------------------------------------------------------------
    target_lags = [1, 2, 3, 4, 5, 6, 12, 18, 36, STEPS_PER_DAY, STEPS_PER_WEEK]
    for lag in target_lags:
        X[f"app_lag{lag}"] = frame[target].shift(lag)

    # --- 3. Rolling window statistics over past values ----------------------------------
    for win, name in [(6, "1h"), (18, "3h"), (36, "6h"), (144, "24h")]:
        roll = past.rolling(win)
        X[f"app_mean_{name}"] = roll.mean()
        X[f"app_std_{name}"] = roll.std()
        X[f"app_max_{name}"] = roll.max()
        X[f"app_min_{name}"] = roll.min()

    if "lights" in frame.columns:
        X["lights_mean_1h"] = frame["lights"].shift(1).rolling(6).mean()
    if "T_out" in frame.columns:
        X["T_out_mean_3h"] = frame["T_out"].shift(1).rolling(18).mean()

    # --- 4. Differences / momentum ------------------------------------------------------
    X["app_diff_10m"] = past - frame[target].shift(2)
    X["app_diff_30m"] = past - frame[target].shift(4)
    X["app_diff_1h"] = past - frame[target].shift(7)
    X["app_diff_24h"] = past - frame[target].shift(STEPS_PER_DAY + 1)
    X["app_accel_10m"] = X["app_diff_10m"] - (frame[target].shift(2) - frame[target].shift(3))

    # --- 5. Calendar and cyclical encodings ---------------------------------------------
    idx = frame.index
    X["hour"] = idx.hour
    X["day_of_week"] = idx.dayofweek
    X["month"] = idx.month
    X["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    nsm = idx.hour * 3600 + idx.minute * 60 + idx.second
    X["nsm"] = nsm
    X["hour_sin"] = np.sin(2 * np.pi * nsm / 86400)
    X["hour_cos"] = np.cos(2 * np.pi * nsm / 86400)
    X["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    X["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)

    # --- 6. Domain: Belgian public holidays inside the observed window ------------------
    holidays = pd.to_datetime([
        "2016-01-01", "2016-03-27", "2016-03-28",
        "2016-05-01", "2016-05-05", "2016-05-15", "2016-05-16"
    ])
    X["is_holiday"] = np.isin(idx.normalize(), holidays).astype(int)
    X["is_non_working"] = ((X["is_weekend"] == 1) | (X["is_holiday"] == 1)).astype(int)

    # --- 7. Domain interaction terms ----------------------------------------------------
    t_cols = [c for c in ["T1", "T2", "T3", "T4", "T5", "T7", "T8", "T9"] if c in frame.columns]
    rh_cols = [c for c in ["RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_7", "RH_8", "RH_9"] if c in frame.columns]

    if t_cols:
        X["T_indoor_mean"] = frame[t_cols].shift(1).mean(axis=1)
        X["T_indoor_range"] = frame[t_cols].shift(1).max(axis=1) - frame[t_cols].shift(1).min(axis=1)
    if rh_cols:
        X["RH_indoor_mean"] = frame[rh_cols].shift(1).mean(axis=1)

    if "T_indoor_mean" in X.columns and "RH_indoor_mean" in X.columns:
        X["T_x_RH_indoor"] = X["T_indoor_mean"] * X["RH_indoor_mean"]

    if "T_out" in frame.columns and "RH_out" in frame.columns:
        X["T_out_x_RH_out"] = frame["T_out"].shift(1) * frame["RH_out"].shift(1)

    if "T_indoor_mean" in X.columns and "T_out" in frame.columns:
        X["indoor_minus_outdoor_T"] = X["T_indoor_mean"] - frame["T_out"].shift(1)

    if "T_out" in frame.columns and "Tdewpoint" in frame.columns:
        X["dewpoint_spread"] = frame["T_out"].shift(1) - frame["Tdewpoint"].shift(1)

    X["evening_peak_flag"] = ((idx.hour >= 17) & (idx.hour <= 20)).astype(int)
    X["hour_x_weekend"] = X["hour_sin"] * X["is_weekend"]

    # Align target and drop warm-up rows (where 1-week lag is NaN)
    combined = X.join(frame[target].rename("__y__")).dropna()
    return combined.drop(columns="__y__"), combined["__y__"].rename(target)


def select_features(X_train, y_train, n_top=30, random_state=42):
    """Rank features by Random Forest importance, Ridge RFE, and target correlation.

    Selects features receiving at least 2 votes across the three ranking methods.
    """
    feature_names = list(X_train.columns)

    # 1. Random Forest feature importance
    rf_selector = RandomForestRegressor(
        n_estimators=200, max_depth=18, min_samples_leaf=3,
        n_jobs=-1, random_state=random_state
    )
    rf_selector.fit(X_train, y_train)
    rf_imp = pd.Series(rf_selector.feature_importances_, index=feature_names).sort_values(ascending=False)

    # 2. RFE with Ridge
    rfe = RFE(Ridge(alpha=1.0), n_features_to_select=n_top, step=5).fit(X_train, y_train)
    rfe_keep = set(X_train.columns[rfe.support_])

    # 3. Correlation with target
    y_series = y_train if isinstance(y_train, pd.Series) else pd.Series(y_train, index=X_train.index)
    corr_rank = X_train.corrwith(y_series).abs().sort_values(ascending=False)

    # Voting tally
    votes = pd.Series(0, index=feature_names)
    votes[rf_imp.head(n_top).index] += 1
    votes[list(rfe_keep)] += 1
    votes[corr_rank.head(n_top).index] += 1

    selected = sorted(votes[votes >= 2].index.tolist())
    return selected, rf_imp, corr_rank

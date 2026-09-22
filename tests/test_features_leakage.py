"""The leakage suite.

These are the most valuable tests in the project. Temporal leakage does not raise an error, it
silently improves your validation score and then destroys your production performance, so it
has to be tested for rather than reviewed for.

The central test is empirical rather than structural: corrupt every observation from some
timestamp onward, rebuild the features, and assert that no feature row at or before that
timestamp moved. If any feature peeks into the future, its value changes and the test fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_forecast.exceptions import LeakageError
from energy_forecast.features import Availability, Family, FeatureBuilder


def test_future_corruption_does_not_change_past_features(synthetic_frame, synthetic_config):
    """No feature may depend on data at or after the row it describes.

    This catches the classic mistake - ``rolling(w).mean()`` without a prior ``shift`` - which
    a code review routinely misses because the line looks entirely reasonable.
    """
    builder = FeatureBuilder(synthetic_config)
    baseline, _ = builder.build(synthetic_frame)

    cut = len(synthetic_frame) // 2
    cut_timestamp = synthetic_frame.index[cut]

    corrupted_frame = synthetic_frame.copy()
    numeric = corrupted_frame.select_dtypes(include=[np.number]).columns
    corrupted_frame.loc[cut_timestamp:, numeric] *= -7.3
    corrupted_frame.loc[cut_timestamp:, numeric] += 1234.0

    corrupted, _ = FeatureBuilder(synthetic_config).build(corrupted_frame)

    common = baseline.index.intersection(corrupted.index)
    unaffected = common[common <= cut_timestamp]
    assert len(unaffected) > 100, "not enough pre-cut rows to make the test meaningful"

    pd.testing.assert_frame_equal(
        baseline.loc[unaffected], corrupted.loc[unaffected],
        check_exact=False, atol=1e-9,
        obj="features at or before the corruption point",
    )


def test_target_is_never_a_feature(synthetic_frame, synthetic_config):
    """The unlagged target must not appear in the design matrix under any name."""
    builder = FeatureBuilder(synthetic_config)
    X, y = builder.build(synthetic_frame)

    assert synthetic_config.target not in X.columns

    # A feature identical to the target would be leakage wearing a different label.
    identical = [c for c in X.columns if np.allclose(X[c].to_numpy(), y.to_numpy())]
    assert not identical, f"features identical to the target: {identical}"


def test_no_feature_correlates_perfectly_with_the_target(synthetic_frame, synthetic_config):
    """A correlation of ~1.0 against the target is the signature of a leaked column."""
    X, y = FeatureBuilder(synthetic_config).build(synthetic_frame)
    correlation = X.corrwith(y).abs()
    suspicious = correlation[correlation > 0.999]
    assert suspicious.empty, f"suspiciously perfect correlations: {suspicious.to_dict()}"


def test_audit_marks_every_observed_feature_as_lagged(synthetic_frame, synthetic_config):
    """Every observed feature must declare a lag of at least the horizon."""
    builder = FeatureBuilder(synthetic_config)
    builder.build(synthetic_frame)
    audit = builder.audit()

    horizon = int(synthetic_config.features["horizon"])
    observed = audit[audit["availability"] == Availability.LAGGED_OBSERVATION.value]
    assert not observed.empty
    assert (observed["min_lag_steps"] >= horizon).all()

    known = audit[audit["availability"] == Availability.KNOWN_IN_ADVANCE.value]
    assert set(known.index) >= {"hour", "is_weekend", "is_holiday"}


def test_builder_rejects_an_unlagged_observed_feature(synthetic_config):
    """The guard in ``_record`` must reject a feature that breaks the rule."""
    builder = FeatureBuilder(synthetic_config)
    with pytest.raises(LeakageError, match="may not use data from the row it predicts"):
        builder._record("bad_feature", Family.EXOGENOUS, Availability.LAGGED_OBSERVATION,
                        min_lag=0, description="deliberately invalid")


def test_rolling_features_exclude_the_current_row(synthetic_frame, synthetic_config):
    """A rolling mean must equal the mean of the *previous* window, not one including t."""
    X, _ = FeatureBuilder(synthetic_config).build(synthetic_frame)
    target = synthetic_frame[synthetic_config.target]

    timestamp = X.index[500]
    position = synthetic_frame.index.get_loc(timestamp)
    horizon = int(synthetic_config.features["horizon"])
    width = int(synthetic_config.features["rolling_windows"]["1h"])

    expected = target.iloc[position - horizon - width + 1: position - horizon + 1].mean()
    assert X.loc[timestamp, "app_mean_1h"] == pytest.approx(expected)

"""Chronological splitting and training-only preprocessing.

The second leakage surface after feature engineering: a scaler fitted on all the data has been
told something about the period it will be scored on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_forecast.features import FeatureBuilder
from energy_forecast.features.preprocessing import Preprocessor, TargetTransformer, Winsoriser
from energy_forecast.data.splitting import chronological_split


@pytest.fixture()
def design(synthetic_frame, synthetic_config):
    """Design matrix and target for the synthetic series."""
    return FeatureBuilder(synthetic_config).build(synthetic_frame)


def test_split_is_strictly_ordered_in_time(design, synthetic_config):
    """train < validation < test, with no overlap and nothing shuffled."""
    X, y = design
    split = chronological_split(X, y, synthetic_config)

    assert split.X_train.index.max() < split.X_val.index.min()
    assert split.X_val.index.max() < split.X_test.index.min()
    assert len(split.X_train) + len(split.X_val) + len(split.X_test) == len(X)

    # Row order must be preserved exactly - a shuffle would break this.
    rebuilt = pd.concat([split.X_train, split.X_val, split.X_test])
    pd.testing.assert_index_equal(rebuilt.index, X.index)


def test_split_respects_configured_fractions(design, synthetic_config):
    """The test block is the configured share of the timeline."""
    X, y = design
    split = chronological_split(X, y, synthetic_config)
    expected = float(synthetic_config.split["test_fraction"])
    assert abs(len(split.X_test) / len(X) - expected) < 0.01


def test_split_rejects_misaligned_inputs(design, synthetic_config):
    """X and y must share an index."""
    X, y = design
    with pytest.raises(ValueError, match="identical index"):
        chronological_split(X, y.iloc[:-5], synthetic_config)


def test_winsoriser_fences_come_only_from_training_rows(design, synthetic_config):
    """A fence must not move when unseen data contains a more extreme value."""
    X, y = design
    split = chronological_split(X, y, synthetic_config)

    winsoriser = Winsoriser(3.0).fit(split.X_train)
    fences = dict(winsoriser.bounds)

    # An extreme value in the test block must not change the learned fences.
    polluted = split.X_test.copy()
    polluted.iloc[0] = polluted.iloc[0] * 1000
    winsoriser.transform(polluted)

    assert winsoriser.bounds == fences


def test_winsoriser_leaves_indicators_alone(design, synthetic_config):
    """Clipping a binary flag or a sine wave to an IQR fence is meaningless."""
    X, y = design
    split = chronological_split(X, y, synthetic_config)
    winsorised = Winsoriser(3.0).fit_transform(split.X_train)

    for column in ("is_weekend", "hour_sin", "is_holiday"):
        if column in split.X_train.columns:
            pd.testing.assert_series_equal(winsorised[column], split.X_train[column])


def test_target_transformer_round_trips():
    """inverse(transform(y)) must return the original Wh values."""
    values = np.array([10.0, 50.0, 60.0, 1080.0, 250.0])
    transformer = TargetTransformer(log_transform=True).fit(values)
    np.testing.assert_allclose(transformer.inverse(transformer.transform(values)),
                               values, atol=1e-6)


def test_target_transformer_reduces_skew():
    """The log transform must actually tame the right tail."""
    rng = np.random.default_rng(0)
    skewed = rng.gamma(1.2, 60.0, 5000) + 10
    transformer = TargetTransformer(log_transform=True).fit(skewed)
    assert abs(pd.Series(transformer.transform(skewed)).skew()) < abs(pd.Series(skewed).skew())


def test_preprocessor_rejects_a_changed_schema(design, synthetic_config):
    """Silently reordered or renamed columns must not be tolerated at inference."""
    X, y = design
    split = chronological_split(X, y, synthetic_config)
    preprocessor = Preprocessor(synthetic_config).fit(split.X_train, split.y_train)

    with pytest.raises(ValueError, match="feature columns differ"):
        preprocessor.transform_features(split.X_test.iloc[:, :-3])


def test_preprocessor_round_trips_through_disk(design, synthetic_config, tmp_path):
    """A model without its preprocessing is not reproducible; both must persist together."""
    X, y = design
    split = chronological_split(X, y, synthetic_config)
    preprocessor = Preprocessor(synthetic_config).fit(split.X_train, split.y_train)

    path = preprocessor.save(tmp_path / "preprocessor.joblib")
    restored = Preprocessor.load(path)

    np.testing.assert_allclose(restored.transform_features(split.X_test),
                               preprocessor.transform_features(split.X_test))

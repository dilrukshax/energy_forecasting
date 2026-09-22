"""End-to-end behaviour of the TensorFlow-free stages.

Deliberately excludes the deep models: CI should be able to prove the data, feature, split,
preprocessing and baseline logic is correct in seconds, without installing TensorFlow.
"""

from __future__ import annotations

import numpy as np
import pytest

from energy_forecast.config import Config, load_config
from energy_forecast.exceptions import ConfigurationError
from energy_forecast.evaluation.metrics import compute_metrics
from energy_forecast.training.pipeline import prepare, run_baseline_stage
from energy_forecast.features.selection import select_features


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """Run the preparation stages once against the real dataset, if present."""
    config = load_config()
    if not config.raw_path.is_file():
        pytest.skip(f"raw dataset not present at {config.raw_path}")
    return prepare(config)


@pytest.mark.slow
def test_prepare_produces_aligned_artifacts(prepared):
    """Shapes and indices must line up across every artefact the run produces."""
    assert prepared.X.index.equals(prepared.y.index)
    assert prepared.X_train_sel.shape[0] == len(prepared.split.X_train)
    assert prepared.X_test_sel.shape[0] == len(prepared.split.X_test)
    assert prepared.X_train_sel.shape[1] == len(prepared.selection.selected)
    assert len(prepared.audit) == prepared.X.shape[1]


@pytest.mark.slow
def test_scaled_training_features_are_standardised(prepared):
    """The scaler is fitted on train, so train should be centred - and test need not be."""
    assert np.allclose(prepared.X_train_sel.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(prepared.X_train_sel.std(axis=0), 1, atol=1e-6)


@pytest.mark.slow
def test_baselines_beat_a_constant_mean(prepared):
    """A model worse than predicting the mean is broken, not merely weak."""
    artifacts = run_baseline_stage(prepared)
    actual = artifacts.split.y_test.to_numpy(dtype=float)
    naive = compute_metrics(actual, np.full_like(actual, actual.mean()), "mean")

    for record in artifacts.metrics:
        assert record.mae < naive.mae, f"{record.model} is worse than a constant mean"


@pytest.mark.slow
def test_persistence_is_a_genuine_competitor(prepared):
    """Documented expectation: at a 10-minute horizon persistence is hard to beat.

    Pinned as a test because if a model ever beats it by an implausible margin, leakage is the
    first thing to suspect.
    """
    artifacts = run_baseline_stage(prepared)
    table = artifacts.comparison
    persistence = table.loc["Persistence", "mae"]
    best = table["mae"].min()
    assert best <= persistence
    assert best > persistence * 0.5, "an implausible improvement over persistence suggests leakage"


def test_selection_requires_a_majority(synthetic_frame, synthetic_config):
    """With an impossible vote threshold, selection must fail loudly rather than return junk."""
    from energy_forecast.features import FeatureBuilder
    from energy_forecast.data.splitting import chronological_split

    X, y = FeatureBuilder(synthetic_config).build(synthetic_frame)
    split = chronological_split(X, y, synthetic_config)

    impossible = Config(**{**{k: getattr(synthetic_config, k)
                              for k in Config.__dataclass_fields__},
                           "selection": {**synthetic_config.selection, "min_votes": 4}})
    with pytest.raises(ConfigurationError, match="retained nothing"):
        select_features(split.X_train, np.log1p(split.y_train.to_numpy()),
                        split.y_train, impossible)

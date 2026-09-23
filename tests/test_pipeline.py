"""End-to-end behaviour of the TensorFlow-free stages.

Deliberately excludes the deep models: CI should be able to prove the data, feature, split,
preprocessing and baseline logic is correct in seconds, without installing TensorFlow.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from energy_forecast.config import Config, load_config
from energy_forecast.evaluation.metrics import compute_metrics
from energy_forecast.exceptions import ConfigurationError
from energy_forecast.features.selection import select_features
from energy_forecast.training.pipeline import finalize_run, prepare, run_baseline_stage


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
    from energy_forecast.data.splitting import chronological_split
    from energy_forecast.features import FeatureBuilder

    X, y = FeatureBuilder(synthetic_config).build(synthetic_frame)
    split = chronological_split(X, y, synthetic_config)

    impossible = Config(**{**{k: getattr(synthetic_config, k)
                              for k in Config.__dataclass_fields__},
                           "selection": {**synthetic_config.selection, "min_votes": 4}})
    with pytest.raises(ConfigurationError, match="retained nothing"):
        select_features(split.X_train, np.log1p(split.y_train.to_numpy()),
                        split.y_train, impossible)


@pytest.mark.slow
def test_saved_model_is_chosen_on_validation_and_serves_aligned_rows(prepared, tmp_path):
    """Test scores must not choose the model, and the saved model must score new timestamps."""
    import copy
    import json

    import pandas as pd

    from energy_forecast.prediction import ForecastService

    artifacts = copy.copy(prepared)
    artifacts.predictions = {}
    artifacts.metrics = []
    artifacts.validation_metrics = {}
    artifacts.fitted_models = {}
    artifacts = run_baseline_stage(artifacts)
    assert artifacts.comparison.index[0] != "Persistence"
    artifacts.validation_metrics["Persistence"] = replace(
        artifacts.validation_metrics["Persistence"], mae=0.0)

    scoped = Config(**{**{k: getattr(prepared.config, k) for k in Config.__dataclass_fields__},
                       "outputs": {**prepared.config.outputs,
                                   "models_dir": str(tmp_path / "models"),
                                   "experiments_dir": str(tmp_path / "experiments")}})
    artifacts.config = scoped
    finalize_run(artifacts)
    assert artifacts.selected_model == "Persistence"
    metadata = json.loads((tmp_path / "models" / "model_metadata.json").read_text())
    assert metadata["kind"] == "persistence"

    service = ForecastService.load(scoped)
    future = artifacts.frame.index[-1] + pd.Timedelta(scoped.frequency)
    future_frame = pd.concat([
        artifacts.frame,
        pd.DataFrame(np.nan, index=pd.DatetimeIndex([future]),
                     columns=artifacts.frame.columns),
    ])
    predicted = service.predict_frame(future_frame)
    assert predicted.index[-1] == future
    assert predicted.loc[future] == artifacts.frame[scoped.target].iloc[-1]


@pytest.mark.slow
def test_windowed_serving_uses_the_last_row_of_each_window(prepared):
    """A Keras prediction must be indexed by its actual target timestamp."""
    import pandas as pd

    from energy_forecast.features import FeatureBuilder
    from energy_forecast.prediction import ForecastService

    class DummyModel:
        def predict(self, windows, verbose=0):
            return np.zeros((len(windows), 1))

    history = prepared.frame.tail(1100)
    future = history.index[-1] + pd.Timedelta(prepared.config.frequency)
    frame = pd.concat([
        history,
        pd.DataFrame(np.nan, index=pd.DatetimeIndex([future]), columns=history.columns),
    ])
    X, _ = FeatureBuilder(prepared.config).build(frame, require_target=False)
    service = ForecastService(prepared.config, DummyModel(), prepared.preprocessor,
                              prepared.selection.selected, "keras")
    result = service.predict_frame(frame)
    expected = X.index[int(prepared.config.sequences["lookback"]) - 1:]
    assert result.index.equals(expected)
    assert result.index[-1] == future

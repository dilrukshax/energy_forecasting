"""The forecaster contract, and run provenance."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from energy_forecast.exceptions import EnergyForecastError, LeakageError, NotFittedError
from energy_forecast.features.preprocessing import TargetTransformer
from energy_forecast.models.base import Forecaster, PersistenceForecaster, SklearnForecaster
from energy_forecast.utils.tracking import (
    compare_runs,
    config_digest,
    environment,
    file_digest,
    git_state,
    new_run,
    save_run,
)


# -- the contract -------------------------------------------------------------------------
def test_forecaster_cannot_be_instantiated_directly():
    """The ABC exists to force implementations to provide fit and predict."""
    with pytest.raises(TypeError):
        Forecaster()  # type: ignore[abstract]


def test_predict_before_fit_raises():
    """Using an unfitted model must fail loudly rather than return nonsense."""
    model = SklearnForecaster(Ridge(), TargetTransformer().fit([10.0, 20.0]), "ridge")
    with pytest.raises(NotFittedError):
        model.predict(np.zeros((3, 2)))


def test_persistence_returns_the_lagged_column():
    """Persistence is the reference every other model is judged against."""
    frame = pd.DataFrame({"app_lag1": [10.0, 20.0, 30.0], "other": [1, 2, 3]})
    model = PersistenceForecaster().fit(frame)
    np.testing.assert_array_equal(model.predict(frame), [10.0, 20.0, 30.0])
    assert model.is_fitted


def test_persistence_rejects_a_missing_lag_column():
    """A silently absent lag column would make the benchmark meaningless."""
    with pytest.raises(KeyError):
        PersistenceForecaster().fit(pd.DataFrame({"unrelated": [1, 2]}))


def test_sklearn_adapter_returns_wh_not_model_space():
    """The adapter owns the inverse transform, so callers never see half-transformed values."""
    wh = np.array([50.0, 60.0, 70.0, 80.0, 400.0])
    transformer = TargetTransformer(log_transform=True).fit(wh)

    X = np.arange(10, dtype=float).reshape(5, 2)
    model = SklearnForecaster(Ridge(alpha=1e-8), transformer, "ridge")
    model.fit(X, transformer.transform(wh))

    predicted = model.predict(X)

    # The contract is the inverse transform, not accuracy: a two-feature ridge cannot fit five
    # arbitrary points, and asserting that it does would be testing the wrong thing.
    np.testing.assert_allclose(
        predicted, transformer.inverse(model.estimator.predict(X)))
    assert predicted.min() > 0, "predictions must be on the Wh scale, not in log space"
    assert 1 < predicted.mean() < 10_000, "predictions are not on a plausible Wh scale"


def test_every_model_satisfies_the_same_interface():
    """The pipeline relies on being able to treat all models identically."""
    transformer = TargetTransformer().fit([10.0, 100.0])
    for model in (PersistenceForecaster(),
                  SklearnForecaster(Ridge(), transformer, "ridge")):
        assert isinstance(model, Forecaster)
        assert hasattr(model, "fit") and hasattr(model, "predict")
        assert model.is_fitted is False
        assert model.name


# -- exceptions ---------------------------------------------------------------------------
def test_every_package_error_shares_one_root():
    """A caller can catch this package's failures without swallowing unrelated ones."""
    for error in (LeakageError, NotFittedError):
        assert issubclass(error, EnergyForecastError)
    assert not issubclass(EnergyForecastError, ValueError)


# -- tracking -----------------------------------------------------------------------------
def test_file_digest_is_stable_and_content_sensitive(tmp_path):
    """The data digest must change when the data changes, and not otherwise."""
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    first = file_digest(path)

    assert first == file_digest(path)

    path.write_text("a,b\n1,3\n", encoding="utf-8")
    assert file_digest(path) != first


def test_config_digest_changes_with_the_config(config):
    """Two runs with the same digests should be reproducible; a change must be visible."""
    from energy_forecast.config import Config

    baseline = config_digest(config)
    assert baseline == config_digest(config)

    altered = Config(**{**{k: getattr(config, k) for k in Config.__dataclass_fields__},
                        "sequences": {"lookback": 999}})
    assert config_digest(altered) != baseline


def test_environment_records_the_libraries_a_result_depends_on():
    """A result is only reproducible if the stack that produced it is recorded."""
    versions = environment()
    assert "python" in versions
    for library in ("numpy", "pandas", "sklearn", "tensorflow"):
        assert library in versions  # "not installed" is a valid, informative value


def test_git_state_degrades_gracefully():
    """Outside a checkout the fields are None rather than an exception."""
    state = git_state()
    assert set(state) == {"commit", "branch", "dirty"}


def test_manifest_round_trips_through_disk(config, tmp_path):
    """The manifest is the audit record; it has to survive serialisation."""
    from energy_forecast.config import Config

    # Manifests are written to the experiments directory, so that is what the test redirects.
    scoped = Config(**{**{k: getattr(config, k) for k in Config.__dataclass_fields__},
                       "outputs": {**config.outputs, "experiments_dir": str(tmp_path)}})

    manifest = new_run(scoped, notes="unit test")
    manifest.best_model = "gru"
    manifest.metrics = [{"model": "gru", "mae": 21.5}]

    path = save_run(manifest, scoped)
    assert path.exists()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["notes"] == "unit test"
    assert payload["best_model"] == "gru"
    assert payload["finished_at"] is not None
    assert (tmp_path / "runs" / "latest.json").exists()


def test_compare_runs_ranks_recorded_runs(config, tmp_path):
    """Comparing a retrain against what is deployed should be reading files, not remembering."""
    from energy_forecast.config import Config

    scoped = Config(**{**{k: getattr(config, k) for k in Config.__dataclass_fields__},
                       "outputs": {**config.outputs, "experiments_dir": str(tmp_path)}})

    for name, mae in (("worse", 30.0), ("better", 20.0)):
        manifest = new_run(scoped)
        manifest.run_id = name
        manifest.best_model = "gru"
        manifest.metrics = [{"model": "gru", "mae": mae}]
        save_run(manifest, scoped)

    table = compare_runs(tmp_path)
    assert list(table.index) == ["better", "worse"]
    assert table.loc["better", "mae"] == 20.0


def test_compare_runs_handles_an_empty_history(tmp_path):
    """A fresh checkout has no runs; that is not an error."""
    assert compare_runs(tmp_path).empty


def test_manifests_fall_back_to_the_reports_directory(config, tmp_path):
    """With no experiments directory configured, runs still get recorded somewhere."""
    from energy_forecast.config import Config

    outputs = {k: v for k, v in config.outputs.items() if k != "experiments_dir"}
    scoped = Config(**{**{k: getattr(config, k) for k in Config.__dataclass_fields__},
                       "outputs": {**outputs, "reports_dir": str(tmp_path)}})

    path = save_run(new_run(scoped), scoped)
    assert path.is_relative_to(tmp_path)

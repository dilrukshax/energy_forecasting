"""End-to-end orchestration.

This is the only module that knows the order of the stages. Everything it calls is independently
testable, which is the whole point of the layering: the leakage rules live in
:mod:`features`, not in a 400-line script that cannot be exercised piecemeal.

The same functions back both the batch run (``energy-forecast train``) and the report notebook,
so the notebook cannot drift away from what actually runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from energy_forecast.models.baselines import run_baselines
from energy_forecast.config import Config, load_config
from energy_forecast.data import DataQualityReport, build_dataset
from energy_forecast.features import FeatureBuilder
from energy_forecast.utils.logging import get_logger
from energy_forecast.evaluation.metrics import Metrics, compute_metrics, metrics_table
from energy_forecast.features.preprocessing import Preprocessor
from energy_forecast.exceptions import NotFittedError
from energy_forecast.features.selection import SelectionResult, select_features
from energy_forecast.training.sequences import SequenceData, build_sequence_data
from energy_forecast.data.splitting import Split, chronological_split
from energy_forecast.utils.tracking import RunManifest, new_run, save_run

logger = get_logger(__name__)


@dataclass
class PipelineArtifacts:
    """Everything a run produces, so the notebook can render it without recomputation.

    Attributes:
        config: The configuration the run used.
        frame: The regularised dataset.
        quality: Data quality report.
        X, y: Full design matrix and target.
        audit: Serving-time availability audit, one row per feature.
        split: Chronological partition.
        preprocessor: Fitted winsoriser, feature scaler and target transformer.
        selection: Feature selection outcome.
        sequences: Windowed tensors, populated only when the deep stage runs.
        predictions: Test predictions in Wh, keyed by model name.
        metrics: One record per model.
        histories: Keras training histories, keyed by model name.
        trials: Hyper-parameter search records.
    """

    config: Config
    frame: pd.DataFrame
    quality: DataQualityReport
    X: pd.DataFrame
    y: pd.Series
    audit: pd.DataFrame
    split: Split
    preprocessor: Preprocessor
    selection: SelectionResult
    sequences: Optional[SequenceData] = None
    predictions: Dict[str, np.ndarray] = field(default_factory=dict)
    metrics: List[Metrics] = field(default_factory=list)
    histories: Dict[str, Any] = field(default_factory=dict)
    trials: List[Any] = field(default_factory=list)
    manifest: Optional[RunManifest] = None

    # Scaled, selected matrices, kept for the modelling stages.
    X_train_sel: Optional[np.ndarray] = field(default=None, repr=False)
    X_val_sel: Optional[np.ndarray] = field(default=None, repr=False)
    X_test_sel: Optional[np.ndarray] = field(default=None, repr=False)
    y_train_scaled: Optional[np.ndarray] = field(default=None, repr=False)
    y_val_scaled: Optional[np.ndarray] = field(default=None, repr=False)
    y_test_scaled: Optional[np.ndarray] = field(default=None, repr=False)

    def require_matrices(self) -> Tuple[np.ndarray, ...]:
        """Return the scaled, selected matrices, asserting the preparation stage has run.

        The fields are optional because the dataclass is constructed incrementally, but every
        modelling stage needs them. Without this guard, calling a stage out of order fails
        somewhere deep in NumPy instead of saying what actually went wrong.
        """
        values = (self.X_train_sel, self.X_val_sel, self.X_test_sel,
                  self.y_train_scaled, self.y_val_scaled, self.y_test_scaled)
        if any(value is None for value in values):
            raise NotFittedError("prepare() must run before the modelling stages")
        return cast(Tuple[np.ndarray, ...], values)

    @property
    def comparison(self) -> pd.DataFrame:
        """Metric table across every model scored so far."""
        return metrics_table(self.metrics)

    @property
    def test_actual(self) -> np.ndarray:
        """Observed test values in Wh, on the rows the deep models also cover."""
        if self.sequences is not None:
            return self.preprocessor.target_transformer.inverse(self.sequences.y_test)
        return self.split.y_test.to_numpy(dtype=float)


def prepare(config: Optional[Config] = None,
            data_path: Optional[Path] = None) -> PipelineArtifacts:
    """Run every stage up to and including feature selection.

    This is the deterministic, TensorFlow-free part of the pipeline: load, validate, engineer,
    split, fit preprocessing on train only, select features.

    Args:
        config: Loaded configuration. Read from disk when omitted.
        data_path: Optional override for the raw CSV.

    Returns:
        Populated :class:`PipelineArtifacts`.
    """
    config = config or load_config()
    manifest = new_run(config)

    frame, quality = build_dataset(config, data_path)

    builder = FeatureBuilder(config)
    X, y = builder.build(frame)
    audit = builder.audit()

    split = chronological_split(X, y, config)

    preprocessor = Preprocessor(config).fit(split.X_train, split.y_train)
    X_train = preprocessor.transform_features(split.X_train)
    X_val = preprocessor.transform_features(split.X_val)
    X_test = preprocessor.transform_features(split.X_test)

    transformer = preprocessor.target_transformer
    y_train = transformer.transform(split.y_train)
    y_val = transformer.transform(split.y_val)
    y_test = transformer.transform(split.y_test)

    selection = select_features(split.X_train, y_train, split.y_train, config)

    artifacts = PipelineArtifacts(
        config=config, frame=frame, quality=quality, X=X, y=y, audit=audit,
        split=split, preprocessor=preprocessor, selection=selection,
        X_train_sel=X_train[:, selection.indices],
        X_val_sel=X_val[:, selection.indices],
        X_test_sel=X_test[:, selection.indices],
        y_train_scaled=y_train, y_val_scaled=y_val, y_test_scaled=y_test,
        manifest=manifest,
    )
    manifest.data_quality = quality.to_dict()
    manifest.selected_features = selection.selected
    logger.info("preparation complete: %s features selected", len(selection.selected))
    return artifacts


def run_baseline_stage(artifacts: PipelineArtifacts) -> PipelineArtifacts:
    """Fit the reference models and score them on the test block."""
    X_train, _, X_test, y_train, _, _ = artifacts.require_matrices()
    predictions = run_baselines(
        X_train, y_train, X_test,
        artifacts.split.X_test, artifacts.preprocessor.target_transformer, artifacts.config,
    )
    actual = artifacts.split.y_test.to_numpy(dtype=float)
    for name, values in predictions.items():
        artifacts.predictions[name] = values
        artifacts.metrics.append(compute_metrics(actual, values, name))
    return artifacts


def run_deep_stage(artifacts: PipelineArtifacts, verbose: int = 0) -> PipelineArtifacts:
    """Window the data, train every configured architecture, and score them.

    Baseline metrics are recomputed on the windowed test rows so that every model in the final
    table is scored on an identical set of timestamps.
    """
    from energy_forecast.models.architectures import Hyperparameters, build_model, set_seeds, train

    config = artifacts.config
    set_seeds(config)

    X_train, X_val, X_test, y_train, y_val, y_test = artifacts.require_matrices()
    sequences = build_sequence_data(
        X_train, X_val, X_test, y_train, y_val, y_test,
        artifacts.split, artifacts.X.index, int(config.sequences["lookback"]),
    )
    artifacts.sequences = sequences
    actual = artifacts.test_actual

    # Re-score the baselines on the common window so the comparison is like for like.
    artifacts.metrics = [
        compute_metrics(actual, values[-len(actual):], name)
        for name, values in artifacts.predictions.items()
    ]

    defaults = config.training["defaults"]
    params = Hyperparameters(units=int(defaults["units"]), dropout=float(defaults["dropout"]),
                             learning_rate=float(defaults["learning_rate"]),
                             batch_size=int(config.training["batch_size"]))
    transformer = artifacts.preprocessor.target_transformer

    for architecture in config.training["architectures"]:
        logger.info("training %s", architecture)
        model = build_model(architecture, sequences.input_shape, params)
        history = train(model, sequences.X_train, sequences.y_train,
                        sequences.X_val, sequences.y_val, params,
                        epochs=int(config.training["epochs"]),
                        early_stopping_patience=int(config.training["early_stopping_patience"]),
                        reduce_lr_patience=int(config.training["reduce_lr_patience"]),
                        verbose=verbose)
        predicted = transformer.inverse(model.predict(sequences.X_test, verbose=0).ravel())

        artifacts.histories[architecture] = history
        artifacts.predictions[architecture] = predicted
        artifacts.metrics.append(compute_metrics(actual, predicted, architecture))
        logger.info("%s scored: MAE %.2f Wh", architecture, artifacts.metrics[-1].mae)

    return artifacts


def run_tuning_stage(artifacts: PipelineArtifacts, verbose: int = 0) -> PipelineArtifacts:
    """Random-search the best recurrent architecture, then re-score it on the test block."""
    from energy_forecast.models.architectures import Hyperparameters, build_model, save_model, train
    from energy_forecast.training.tuning import random_search

    config = artifacts.config
    if not config.tuning.get("enabled", True):
        logger.info("tuning disabled in config; skipping")
        return artifacts
    if artifacts.sequences is None:
        raise RuntimeError("run_deep_stage must run before run_tuning_stage")

    table = artifacts.comparison
    candidates = [m for m in table.index if m in config.training["architectures"]]
    if not candidates:
        logger.warning("no deep model to tune; skipping")
        return artifacts
    architecture = table.loc[candidates, "mae"].idxmin()
    logger.info("tuning %s", architecture)

    trials = random_search(architecture, artifacts.sequences,
                           artifacts.preprocessor.target_transformer, config)
    artifacts.trials = trials
    best = trials[0].params

    from energy_forecast.models.architectures import _keras

    keras = _keras()
    keras.backend.clear_session()
    keras.utils.set_random_seed(config.random_state)

    model = build_model(architecture, artifacts.sequences.input_shape, best)
    history = train(model, artifacts.sequences.X_train, artifacts.sequences.y_train,
                    artifacts.sequences.X_val, artifacts.sequences.y_val, best,
                    epochs=int(config.training["epochs"]) + 20,
                    early_stopping_patience=int(config.training["early_stopping_patience"]) + 2,
                    reduce_lr_patience=int(config.training["reduce_lr_patience"]),
                    verbose=verbose)

    label = f"{architecture}_tuned"
    transformer = artifacts.preprocessor.target_transformer
    predicted = transformer.inverse(model.predict(artifacts.sequences.X_test, verbose=0).ravel())

    artifacts.histories[label] = history
    artifacts.predictions[label] = predicted
    artifacts.metrics.append(compute_metrics(artifacts.test_actual, predicted, label))

    models_dir = config.output_dir("models_dir")
    save_model(model, models_dir / "best_model.keras")
    artifacts.preprocessor.save(models_dir / "preprocessor.joblib")

    # The feature order is part of the model contract: inference must slice the scaled matrix
    # in exactly the order training used, so it is persisted with the other artefacts.
    with open(models_dir / "selected_features.json", "w", encoding="utf-8") as handle:
        json.dump(artifacts.selection.selected, handle, indent=2)

    return artifacts


def run_all(config: Optional[Config] = None, data_path: Optional[Path] = None,
            skip_deep: bool = False, verbose: int = 0) -> PipelineArtifacts:
    """Run the whole pipeline.

    Args:
        config: Loaded configuration. Read from disk when omitted.
        data_path: Optional override for the raw CSV.
        skip_deep: Run only the deterministic stages and the baselines. Useful in CI, where
            installing TensorFlow to prove the feature logic works is not worth the minutes.
        verbose: Keras verbosity.

    Returns:
        Populated :class:`PipelineArtifacts`.
    """
    artifacts = run_baseline_stage(prepare(config, data_path))
    if not skip_deep:
        artifacts = run_tuning_stage(run_deep_stage(artifacts, verbose=verbose),
                                     verbose=verbose)
    else:
        logger.info("skipping the deep-learning stages (skip_deep=True)")

    if artifacts.manifest is not None:
        artifacts.manifest.metrics = [record.to_dict() for record in artifacts.metrics]
        artifacts.manifest.trials = [trial.to_dict() for trial in artifacts.trials]
        artifacts.manifest.best_model = (artifacts.comparison.index[0]
                                         if artifacts.metrics else None)
        save_run(artifacts.manifest, artifacts.config)
    return artifacts

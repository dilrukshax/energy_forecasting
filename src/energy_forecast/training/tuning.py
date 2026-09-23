"""Hyper-parameter search.

Random search rather than an exhaustive grid: at a budget of a handful of fits, random sampling
covers each individual dimension far better than a coarse grid does, and a full grid over five
hyper-parameters is not affordable for a recurrent model.

Selection happens on the **validation** block. The test block is scored once, afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from energy_forecast.config import Config
from energy_forecast.features.preprocessing import TargetTransformer
from energy_forecast.models.architectures import Hyperparameters, build_model, train
from energy_forecast.training.sequences import SequenceData
from energy_forecast.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Trial:
    """One evaluated configuration."""

    params: Hyperparameters
    val_mae: float
    val_loss: float
    epochs: int

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, JSON-serialisable record."""
        return {**self.params.to_dict(), "val_mae_wh": self.val_mae,
                "val_loss": self.val_loss, "epochs": self.epochs}


def sample_params(space: dict[str, list[Any]], rng: np.random.Generator) -> Hyperparameters:
    """Draw one configuration uniformly from the search space."""
    return Hyperparameters(
        units=int(rng.choice(space["units"])),
        dropout=float(rng.choice(space["dropout"])),
        learning_rate=float(rng.choice(space["learning_rate"])),
        batch_size=int(rng.choice(space["batch_size"])),
        optimizer=str(rng.choice(space["optimizer"])),
    )


def random_search(architecture: str, data: SequenceData, transformer: TargetTransformer,
                  config: Config) -> list[Trial]:
    """Evaluate ``tuning.n_trials`` random configurations on the validation block.

    Args:
        architecture: Name of the architecture to tune.
        data: Windowed tensors.
        transformer: Fitted target transformer, so validation MAE is reported in Wh.
        config: Loaded configuration.

    Returns:
        Trials sorted best-first by validation MAE.

    """
    from energy_forecast.models.architectures import _keras

    keras = _keras()
    settings = config.tuning
    space = settings["search_space"]
    rng = np.random.default_rng(config.random_state)
    val_actual = transformer.inverse(data.y_val)

    trials: list[Trial] = []
    for number in range(int(settings["n_trials"])):
        params = sample_params(space, rng)
        keras.backend.clear_session()
        keras.utils.set_random_seed(config.random_state)

        model = build_model(architecture, data.input_shape, params)
        history = train(model, data.X_train, data.y_train, data.X_val, data.y_val,
                        params, epochs=int(settings["epochs"]),
                        early_stopping_patience=int(settings["early_stopping_patience"]),
                        reduce_lr_patience=int(config.training["reduce_lr_patience"]))

        predicted = transformer.inverse(model.predict(data.X_val, verbose=0).ravel())
        trial = Trial(params=params,
                      val_mae=float(mean_absolute_error(val_actual, predicted)),
                      val_loss=float(min(history.history["val_loss"])),
                      epochs=len(history.history["loss"]))
        trials.append(trial)
        logger.info("trial %s/%s %s -> val MAE %.2f Wh",
                    number + 1, settings["n_trials"], params.to_dict(), trial.val_mae)

    return sorted(trials, key=lambda t: t.val_mae)


def trials_table(trials: list[Trial]) -> pd.DataFrame:
    """Collect trials into a table, best first."""
    return pd.DataFrame([trial.to_dict() for trial in trials]).sort_values("val_mae_wh")

"""Reference models.

The point of a baseline is falsification: a deep model that cannot beat ``y[t] = y[t-1]`` has
learned nothing, however good its R-squared looks. At a 10-minute horizon with a lag-1
autocorrelation around 0.75, persistence is a genuinely strong competitor, so it is reported
first and every other model is judged against it.

Each baseline is returned as a :class:`~energy_forecast.base.Forecaster`, so the pipeline treats
it exactly like a recurrent network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from energy_forecast.config import Config
from energy_forecast.features.preprocessing import TargetTransformer
from energy_forecast.models.base import Forecaster, PersistenceForecaster, SklearnForecaster
from energy_forecast.utils.logging import get_logger

logger = get_logger(__name__)


def build_baselines(transformer: TargetTransformer, config: Config) -> list[Forecaster]:
    """Instantiate the reference models, unfitted.

    Args:
        transformer: Fitted target transformer, so each model can return Wh.
        config: Loaded configuration, for the random seed.

    Returns:
        Persistence, ridge and random forest, in reporting order.

    """
    return [
        PersistenceForecaster(),
        SklearnForecaster(Ridge(alpha=1.0), transformer, "Ridge"),
        SklearnForecaster(
            RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=2,
                                  n_jobs=-1, random_state=config.random_state),
            transformer, "RandomForest",
        ),
    ]


def run_baselines(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
                  X_test: np.ndarray, X_val_frame: pd.DataFrame,
                  X_test_frame: pd.DataFrame, transformer: TargetTransformer,
                  config: Config) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray],
                                           dict[str, Forecaster]]:
    """Fit every baseline and return validation/test predictions and fitted models.

    Persistence reads a column of the unscaled design matrix; the learned models take the
    scaled, selected matrices. That difference is handled here rather than leaking into the
    pipeline.

    Args:
        X_train: Scaled, selected training features.
        y_train: Training target in model space.
        X_val: Scaled, selected validation features.
        X_test: Scaled, selected test features.
        X_val_frame: Unscaled validation design matrix for persistence.
        X_test_frame: Unscaled test design matrix, for the persistence lag column.
        transformer: Fitted target transformer.
        config: Loaded configuration.

    Returns:
        Mapping of model name to predictions in Wh.

    """
    val_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    fitted: dict[str, Forecaster] = {}

    for model in build_baselines(transformer, config):
        if isinstance(model, PersistenceForecaster):
            model.fit(X_test_frame)
            val_predictions[model.name] = model.predict(X_val_frame)
            test_predictions[model.name] = model.predict(X_test_frame)
        else:
            model.fit(X_train, y_train)
            val_predictions[model.name] = model.predict(X_val)
            test_predictions[model.name] = model.predict(X_test)
        fitted[model.name] = model
        logger.info("baseline ready: %s", model.name)

    return val_predictions, test_predictions, fitted

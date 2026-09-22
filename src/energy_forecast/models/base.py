"""The forecaster contract.

Every model in this project - a one-line persistence rule, a scikit-learn regressor and a Keras
recurrent network - is used the same way by the pipeline, the evaluation code and the inference
path. An abstract base class makes that a contract the type checker enforces rather than a
convention each call site re-implements.

The practical payoff: :mod:`pipeline` and :mod:`inference` contain no branching on model type,
and adding a fourth model means implementing three methods, not editing four modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from energy_forecast.exceptions import NotFittedError
from energy_forecast.utils.logging import get_logger
from energy_forecast.features.preprocessing import TargetTransformer

logger = get_logger(__name__)


class Forecaster(ABC):
    """Common interface for every model.

    Implementations differ in what they consume - a flat matrix or a 3-D window tensor - but
    all of them return predictions in **Wh**, so the evaluation code never has to know which
    transform was applied.

    Attributes:
        name: Identifier used in metric tables, figures and the run manifest.
    """

    name: str = "forecaster"

    def __init__(self, name: Optional[str] = None) -> None:
        if name:
            self.name = name
        self._fitted = False

    @abstractmethod
    def fit(self, X: Any, y: Any, **kwargs: Any) -> "Forecaster":
        """Fit on training data. Must set ``self._fitted``."""

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """Return predictions in Wh."""

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has completed."""
        return self._fitted

    def _check_fitted(self) -> None:
        """Raise if the model is used before being fitted."""
        if not self._fitted:
            raise NotFittedError(f"{self.name} must be fitted before predict()")

    def save(self, path: Path) -> Path:  # pragma: no cover - overridden where meaningful
        """Persist the model. Implementations that have nothing to persist may no-op."""
        raise NotImplementedError(f"{type(self).__name__} does not support saving")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, fitted={self._fitted})"


class PersistenceForecaster(Forecaster):
    """Predict the previous observation.

    The benchmark that matters at a 10-minute horizon: with a lag-1 autocorrelation around
    0.75, any model that cannot beat this has learned nothing. It has no parameters, so
    :meth:`fit` only records that it was called.

    Args:
        lag_column: Column of the design matrix holding the one-step lag of the target.
    """

    name = "Persistence"

    def __init__(self, lag_column: str = "app_lag1") -> None:
        super().__init__()
        self.lag_column = lag_column

    def fit(self, X: pd.DataFrame, y: Any = None, **kwargs: Any) -> "PersistenceForecaster":
        """Nothing to learn; validates that the lag column exists."""
        if self.lag_column not in X.columns:
            raise KeyError(f"{self.lag_column!r} not in the design matrix")
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the lagged target, already in Wh."""
        self._check_fitted()
        return X[self.lag_column].to_numpy(dtype=float)


class SklearnForecaster(Forecaster):
    """Adapter for any scikit-learn regressor.

    The estimator is trained in model space (log1p + standardised) and its predictions are
    inverted back to Wh here, so the caller never handles a half-transformed value.

    Args:
        estimator: Any fitted-or-unfitted scikit-learn regressor.
        transformer: Fitted target transformer, for the inverse.
        name: Label for reporting.
    """

    def __init__(self, estimator: Any, transformer: TargetTransformer, name: str) -> None:
        super().__init__(name)
        self.estimator = estimator
        self.transformer = transformer

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "SklearnForecaster":
        """Fit the wrapped estimator on targets already in model space."""
        self.estimator.fit(X, y)
        self._fitted = True
        logger.info("%s fitted on %s rows", self.name, len(X))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict and invert to Wh."""
        self._check_fitted()
        return self.transformer.inverse(self.estimator.predict(X))

    def save(self, path: Path) -> Path:
        """Persist the estimator with joblib."""
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, path)
        return path


class KerasForecaster(Forecaster):
    """Adapter for the recurrent models.

    Consumes 3-D windows rather than flat rows; otherwise identical from the outside, which is
    the point of the contract.

    Args:
        model: A compiled Keras model.
        transformer: Fitted target transformer, for the inverse.
        name: Label for reporting.
    """

    def __init__(self, model: Any, transformer: TargetTransformer, name: str) -> None:
        super().__init__(name)
        self.model = model
        self.transformer = transformer
        self.history: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "KerasForecaster":
        """Fit with early stopping.

        Keyword Args:
            validation_data: ``(X_val, y_val)``, required.
            params: :class:`~energy_forecast.models.Hyperparameters`.
            epochs, early_stopping_patience, reduce_lr_patience, verbose: Passed through.
        """
        from energy_forecast.models.architectures import train

        validation_data = kwargs.pop("validation_data")
        params = kwargs.pop("params")
        self.history = train(self.model, X, y, validation_data[0], validation_data[1],
                             params, **kwargs)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on windows and invert to Wh."""
        self._check_fitted()
        return self.transformer.inverse(self.model.predict(X, verbose=0).ravel())

    def save(self, path: Path) -> Path:
        """Persist in the native Keras format."""
        from energy_forecast.models.architectures import save_model

        return save_model(self.model, path)

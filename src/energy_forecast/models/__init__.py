"""Model definitions.

Every model - a one-line persistence rule, a scikit-learn regressor, a Keras network -
implements the :class:`Forecaster` contract, so the training and prediction layers contain no
branching on model type. TensorFlow is imported lazily by the architectures module.
"""

from energy_forecast.models.base import (
    Forecaster,
    KerasForecaster,
    PersistenceForecaster,
    SklearnForecaster,
)
from energy_forecast.models.baselines import build_baselines, run_baselines

__all__ = [
    "Forecaster", "KerasForecaster", "PersistenceForecaster", "SklearnForecaster",
    "build_baselines", "run_baselines",
]

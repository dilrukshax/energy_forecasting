"""Multivariate time-series forecasting of household appliance energy consumption.

The package is organised by responsibility, so each layer can be tested and replaced in
isolation:

``config``      typed access to ``configs/config.yaml``
``data``        loading, schema validation, time-grid enforcement, chronological splitting
``features``    the leakage-safe design matrix, scaling and feature selection
``models``      the ``Forecaster`` contract, baselines and the Keras architectures
``training``    windowing, stage orchestration and hyper-parameter search
``evaluation``  metrics and figures, always in the original units
``prediction``  inference against saved artefacts
``utils``       structured logging and run provenance
``exceptions``  one error hierarchy for the whole package
``cli``         the command line entry point

Only the leaf modules import TensorFlow, and only inside the functions that need it, so the
correctness-critical layers run and are tested without it.
"""

from energy_forecast import evaluation
from energy_forecast import evaluation as evaluate
from energy_forecast.exceptions import (
    ArtifactError,
    ConfigurationError,
    DataValidationError,
    EnergyForecastError,
    LeakageError,
    NotFittedError,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "EnergyForecastError",
    "ConfigurationError",
    "DataValidationError",
    "LeakageError",
    "NotFittedError",
    "ArtifactError",
    "evaluation",
    "evaluate",
]

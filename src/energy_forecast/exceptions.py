"""Exception hierarchy.

A single root means a caller can catch everything this package raises without also swallowing
unrelated ``ValueError``s from third-party code, and a scheduled job can distinguish "the data
arrived broken" from "the code is broken" - which determines whether to page someone or retry.
"""

from __future__ import annotations


class EnergyForecastError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(EnergyForecastError):
    """The configuration file is missing, malformed or internally inconsistent."""


class DataValidationError(EnergyForecastError):
    """The dataset violates an assumption the pipeline depends on.

    Raised for duplicate timestamps, an irregular sampling interval, a missing target column
    or a schema that no longer matches what the model was trained on. Recoverable by fixing
    the input, not by retrying.
    """


class LeakageError(EnergyForecastError):
    """A feature was declared in a way that would let it see its own target.

    Raised at build time by the feature builder rather than discovered later as an
    implausibly good validation score.
    """


class NotFittedError(EnergyForecastError):
    """A transformer or model was used before being fitted."""


class ArtifactError(EnergyForecastError):
    """A saved model, preprocessor or run artefact is missing or unreadable."""

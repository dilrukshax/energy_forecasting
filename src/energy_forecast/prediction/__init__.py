"""Inference against saved artefacts.

Reuses the training feature builder and the fitted preprocessor, so training/serving skew is
structurally impossible rather than merely avoided by care.
"""

from energy_forecast.prediction.service import ForecastService

__all__ = ["ForecastService"]

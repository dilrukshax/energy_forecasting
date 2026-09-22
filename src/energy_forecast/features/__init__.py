"""Feature engineering, scaling and selection.

The leakage rules live here: every observed input is shifted by at least the forecast horizon,
declared with a ``min_lag``, and rejected at build time if it breaks that contract.
"""

from energy_forecast.features.builder import (
    Availability,
    Family,
    FeatureBuilder,
    FeatureSpec,
    build_features,
)
from energy_forecast.features.preprocessing import (
    Preprocessor,
    TargetTransformer,
    Winsoriser,
    iqr_bounds,
)
from energy_forecast.features.selection import SelectionResult, select_features

__all__ = [
    "Availability", "Family", "FeatureBuilder", "FeatureSpec", "build_features",
    "Preprocessor", "TargetTransformer", "Winsoriser", "iqr_bounds",
    "SelectionResult", "select_features",
]

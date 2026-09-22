"""Data loading, validation and chronological partitioning.

The raw CSV is immutable; everything here is a pure transformation of it, so the analysis-ready
dataset can always be regenerated rather than trusted as an input.
"""

from energy_forecast.data.loader import (
    DataQualityReport,
    build_dataset,
    enforce_regular_grid,
    load_raw,
    longest_constant_run,
    validate,
)
from energy_forecast.data.splitting import Split, chronological_split

__all__ = [
    "DataQualityReport", "build_dataset", "enforce_regular_grid", "load_raw",
    "longest_constant_run", "validate", "Split", "chronological_split",
]

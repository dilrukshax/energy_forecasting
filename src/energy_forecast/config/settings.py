"""Typed configuration loading.

Every tunable lives in ``configs/config.yaml``. Nothing in the package hardcodes a path, a
hyper-parameter or a split fraction, so a run is fully described by its config file - which is
what makes a result reproducible months later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


def project_root() -> Path:
    """Return the repository root, resolved relative to this file.

    Keeps behaviour identical whether the code is invoked from the repo root, from
    ``notebooks/`` or from an installed package. The index is
    ``src/energy_forecast/config/settings.py`` -> four levels up.
    """
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Config:
    """Immutable view over the YAML configuration.

    Attributes mirror the top-level blocks of ``config.yaml``. Paths are resolved against the
    repository root on access so callers never have to care about their working directory.
    """

    project: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)
    split: Dict[str, Any] = field(default_factory=dict)
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    selection: Dict[str, Any] = field(default_factory=dict)
    sequences: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    tuning: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)

    # -- derived accessors ---------------------------------------------------------------
    @property
    def random_state(self) -> int:
        """Seed applied to NumPy, scikit-learn and Keras."""
        return int(self.project.get("random_state", 42))

    @property
    def target(self) -> str:
        """Name of the target column."""
        return str(self.data["target"])

    @property
    def frequency(self) -> str:
        """Pandas offset alias describing the sampling interval."""
        return str(self.data["frequency"])

    def path(self, value: str) -> Path:
        """Resolve a configured path against the repository root."""
        candidate = Path(value)
        return candidate if candidate.is_absolute() else project_root() / candidate

    @property
    def raw_path(self) -> Path:
        """Location of the immutable raw dataset."""
        return self.path(self.data["raw_path"])

    def output_dir(self, key: str) -> Path:
        """Return a configured output directory, creating it if necessary."""
        directory = self.path(self.outputs[key])
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def steps_per(self, unit: str) -> int:
        """Number of sampling steps in ``hour``, ``day`` or ``week``."""
        minutes = int(pd_offset_minutes(self.frequency))
        per_hour = 60 // minutes
        return {"hour": per_hour, "day": per_hour * 24, "week": per_hour * 24 * 7}[unit]


def pd_offset_minutes(freq: str) -> int:
    """Convert a pandas offset alias such as ``10min`` into whole minutes."""
    import pandas as pd

    return int(pd.tseries.frequencies.to_offset(freq).nanos // 60_000_000_000)


def load_config(path: Path | str | None = None) -> Config:
    """Load and validate the YAML configuration.

    Args:
        path: Optional explicit config path. Defaults to ``configs/config.yaml`` under the
            repository root.

    Returns:
        A frozen :class:`Config`.

    Raises:
        FileNotFoundError: if the config file does not exist.
        KeyError: if a required top-level block is missing.
    """
    config_path = Path(path) if path else project_root() / DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")

    with open(config_path, encoding="utf-8") as handle:
        payload: Mapping[str, Any] = yaml.safe_load(handle)

    required: List[str] = ["project", "data", "features", "split", "preprocessing",
                           "selection", "sequences", "training", "outputs"]
    missing = [block for block in required if block not in payload]
    if missing:
        raise KeyError(f"configuration is missing required block(s): {missing}")

    return Config(**{key: payload.get(key, {}) for key in Config.__dataclass_fields__})

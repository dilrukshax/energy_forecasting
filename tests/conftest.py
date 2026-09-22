"""Shared fixtures.

Tests run against a small synthetic series by default, so the suite stays fast and does not
depend on the real CSV being present. Tests that genuinely need the real data are marked and
skip cleanly when it is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_forecast.config import Config, load_config, project_root


@pytest.fixture(scope="session")
def config() -> Config:
    """The real project configuration."""
    return load_config()


@pytest.fixture(scope="session")
def raw_csv_path(config: Config) -> Path:
    """Path to the real dataset, or skip if it has not been downloaded."""
    if not config.raw_path.is_file():
        pytest.skip(f"raw dataset not present at {config.raw_path}")
    return config.raw_path


@pytest.fixture()
def synthetic_frame() -> pd.DataFrame:
    """A small, well-formed stand-in for the real dataset.

    Three weeks at 10-minute resolution: long enough for the 1008-step weekly lag to produce
    usable rows, small enough that the suite runs in seconds.
    """
    periods = 3 * 7 * 144  # three weeks
    index = pd.date_range("2016-01-11 17:00:00", periods=periods, freq="10min")
    rng = np.random.default_rng(0)

    daily = 60 + 40 * np.sin(2 * np.pi * (index.hour * 60 + index.minute) / 1440 - 1.2)
    noise = rng.gamma(2.0, 12.0, periods)
    appliances = np.clip(daily + noise, 10, None).round()

    columns = {
        "Appliances": appliances,
        "lights": rng.choice([0, 0, 0, 10, 20], periods),
        "T_out": 7 + 5 * np.sin(np.arange(periods) / 500) + rng.normal(0, 0.4, periods),
        "RH_out": np.clip(80 + rng.normal(0, 8, periods), 20, 100),
        "Tdewpoint": 3 + rng.normal(0, 1.5, periods),
        "Press_mm_hg": 750 + rng.normal(0, 4, periods),
        "Windspeed": np.abs(rng.normal(4, 2, periods)),
        "Visibility": np.clip(rng.normal(38, 10, periods), 1, 66),
    }
    for i in list(range(1, 6)) + list(range(7, 10)):
        columns[f"T{i}"] = 20 + rng.normal(0, 1.0, periods)
        columns[f"RH_{i}"] = 40 + rng.normal(0, 3.0, periods)
    columns["T6"] = 7 + rng.normal(0, 3.0, periods)
    columns["RH_6"] = np.clip(55 + rng.normal(0, 20, periods), 1, 99.9)

    return pd.DataFrame(columns, index=index).rename_axis("date")


@pytest.fixture()
def synthetic_config(config: Config, synthetic_frame: pd.DataFrame) -> Config:
    """The real config, with the weekly lag trimmed to fit the synthetic series."""
    features = dict(config.features)
    features["target_lags"] = [lag for lag in features["target_lags"] if lag <= 144]
    features["holidays"] = ["2016-01-01"]
    return Config(**{**{k: getattr(config, k) for k in Config.__dataclass_fields__},
                     "features": features})


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Repository root."""
    return project_root()

"""Configuration loading.

Every tunable lives in ``configs/config.yaml``; nothing in the package hardcodes a path, a
hyper-parameter or a split fraction. A run is therefore fully described by its config file.
"""

from energy_forecast.config.settings import (
    DEFAULT_CONFIG_PATH,
    Config,
    load_config,
    project_root,
)

__all__ = ["Config", "load_config", "project_root", "DEFAULT_CONFIG_PATH"]

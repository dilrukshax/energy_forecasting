"""Logging setup.

The package logs rather than prints: a scheduled retrain writes to a file or a log aggregator,
where ``print`` output is either lost or unattributable.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int = logging.INFO, log_file: Optional[Path] = None) -> None:
    """Configure root logging once, for the whole process.

    Args:
        level: Minimum level to emit.
        log_file: Optional file to mirror the stream handler into.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=DATE_FORMAT,
                        handlers=handlers, force=True)
    # TensorFlow is extremely chatty at import and during fit.
    logging.getLogger("tensorflow").setLevel(logging.ERROR)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)

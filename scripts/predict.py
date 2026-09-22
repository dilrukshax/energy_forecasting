#!/usr/bin/env python
"""Score new data with the saved model and preprocessor.

Usage:
    python scripts/predict.py --input data/raw/new_batch.csv --output reports/predictions.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energy_forecast.config import load_config  # noqa: E402
from energy_forecast.exceptions import EnergyForecastError  # noqa: E402
from energy_forecast.prediction import ForecastService  # noqa: E402
from energy_forecast.utils import configure_logging  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True,
                        help="Raw CSV to score, in the training schema")
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to write predictions (stdout when omitted)")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML")
    parser.add_argument("--models-dir", type=Path, default=None,
                        help="Directory holding the saved artefacts")
    parser.add_argument("--debug", action="store_true", help="Debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load the artefacts, score the input, and report. Returns a process exit code."""
    args = parse_args(argv)
    configure_logging(logging.DEBUG if args.debug else logging.INFO)

    try:
        service = ForecastService.load(load_config(args.config), args.models_dir)
        predictions = service.predict_csv(args.input, args.output)
    except EnergyForecastError as exc:
        # An expected, actionable condition: report it without a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.output is None:
        print(predictions.round(1).to_string())
    print(f"\n{len(predictions)} timestamps scored | mean {predictions.mean():.1f} Wh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

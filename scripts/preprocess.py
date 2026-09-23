#!/usr/bin/env python
"""Build the analysis-ready dataset and design matrix from the raw CSV.

Materialises the deterministic part of the pipeline into ``data/processed/`` so that
exploration, and repeated experiments over the same features, do not re-run feature
engineering every time. Training does not depend on the output - it rebuilds from raw, so the
processed files can always be deleted.

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --audit-only     # just print the feature availability audit
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energy_forecast.config import load_config  # noqa: E402
from energy_forecast.data import build_dataset  # noqa: E402
from energy_forecast.features import FeatureBuilder  # noqa: E402
from energy_forecast.utils import configure_logging  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None, help="Config YAML")
    parser.add_argument("--data", type=Path, default=None, help="Override the raw CSV")
    parser.add_argument("--audit-only", action="store_true",
                        help="Print the serving-time feature audit and exit")
    parser.add_argument("--debug", action="store_true", help="Debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write the processed dataset and the feature audit. Returns a process exit code."""
    args = parse_args(argv)
    configure_logging(logging.DEBUG if args.debug else logging.INFO)

    config = load_config(args.config)
    frame, quality = build_dataset(config, args.data)

    builder = FeatureBuilder(config)
    X, y = builder.build(frame)
    audit = builder.audit()

    if args.audit_only:
        print(audit.to_string())
        return 0

    processed = config.path(config.data["processed_dir"])
    processed.mkdir(parents=True, exist_ok=True)

    frame.to_parquet(processed / "dataset.parquet")
    X.to_parquet(processed / "features.parquet")
    y.to_frame().to_parquet(processed / "target.parquet")
    audit.to_csv(processed / "feature_audit.csv")

    print(f"rows            : {quality.n_rows}")
    print(f"design matrix   : {X.shape[0]} x {X.shape[1]}")
    print(f"written to      : {processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

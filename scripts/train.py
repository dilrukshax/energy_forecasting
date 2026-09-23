#!/usr/bin/env python
"""Train the forecasting models.

A thin argparse wrapper over :mod:`energy_forecast.training.pipeline`, for environments where a
plain script is easier to schedule than a console entry point - cron, a Slurm batch file, an
Airflow ``BashOperator``. All the logic lives in the package; this file only parses arguments.

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --skip-deep          # baselines only, no TensorFlow required
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running straight from a checkout, before `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energy_forecast.config import load_config  # noqa: E402
from energy_forecast.evaluation import (  # noqa: E402
    plot_actual_vs_predicted,
    plot_metric_comparison,
    plot_residual_diagnostics,
    plot_training_history,
    save_figure,
    summarise,
    use_headless_backend,
    write_report,
)
from energy_forecast.training import run_all  # noqa: E402
from energy_forecast.utils import configure_logging  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None,
                        help="Config YAML (default: configs/config.yaml)")
    parser.add_argument("--data", type=Path, default=None,
                        help="Override the raw CSV location")
    parser.add_argument("--skip-deep", action="store_true",
                        help="Baselines only; does not require TensorFlow")
    parser.add_argument("--keras-verbose", type=int, default=0,
                        help="Keras fit verbosity (0, 1 or 2)")
    parser.add_argument("--debug", action="store_true", help="Debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline and write every artefact. Returns a process exit code."""
    args = parse_args(argv)
    config = load_config(args.config)

    reports_dir = config.output_dir("reports_dir")
    configure_logging(logging.DEBUG if args.debug else logging.INFO, reports_dir / "run.log")
    use_headless_backend()

    artifacts = run_all(config, args.data, skip_deep=args.skip_deep,
                        verbose=args.keras_verbose)

    figures_dir = config.output_dir("figures_dir")
    actual = artifacts.test_actual
    index = (artifacts.sequences.test_index if artifacts.sequences is not None
             else artifacts.split.X_test.index)

    save_figure(plot_actual_vs_predicted(index, actual, artifacts.predictions),
                figures_dir / "actual_vs_predicted.png")
    save_figure(plot_metric_comparison(artifacts.comparison),
                figures_dir / "metric_comparison.png")

    best = artifacts.selected_model or artifacts.comparison.index[0]
    save_figure(plot_residual_diagnostics(actual, artifacts.predictions[best][-len(actual):],
                                          best),
                figures_dir / "residual_diagnostics.png")
    if artifacts.histories:
        save_figure(plot_training_history(artifacts.histories),
                    figures_dir / "training_history.png")

    write_report(artifacts.metrics, actual, artifacts.predictions,
                 config.path(config.outputs["metrics_file"]),
                 extra={"data_quality": artifacts.quality.to_dict(),
                        "selected_features": artifacts.selection.selected,
                        "trials": [trial.to_dict() for trial in artifacts.trials],
                        "validation_metrics": {
                            name: metric.to_dict()
                            for name, metric in artifacts.validation_metrics.items()}},
                 selected_model=artifacts.selected_model)

    print(artifacts.comparison.round(3).to_string())
    print()
    print(summarise(artifacts.metrics, actual, artifacts.selected_model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

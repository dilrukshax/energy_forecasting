"""Command line entry point.

Every stage is reachable without a notebook, which is what makes the project schedulable:

    energy-forecast validate                  # data quality only
    energy-forecast audit                     # serving-time feature audit
    energy-forecast train --skip-deep         # baselines only, no TensorFlow needed
    energy-forecast train                     # the full run
    energy-forecast predict --input new.csv   # score new data with the saved artefacts
    energy-forecast runs                      # compare every recorded run
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Callable, Optional

import click

from energy_forecast.config import load_config
from energy_forecast.exceptions import EnergyForecastError
from energy_forecast.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def handle_errors(func: "Callable[..., None]") -> "Callable[..., None]":
    """Turn a package error into a clean CLI message instead of a traceback.

    Anything this package raises deliberately is a condition the user can act on, so it is
    reported as a one-line error with a non-zero exit code. Unexpected exceptions still raise
    with a full traceback, because those are bugs and the stack is the useful part.
    """
    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> None:
        try:
            func(*args, **kwargs)
        except EnergyForecastError as exc:
            raise click.ClickException(str(exc)) from exc

    return wrapper

config_option = click.option(
    "--config", "config_path", type=click.Path(path_type=Path), default=None,
    help="Path to a config YAML. Defaults to configs/config.yaml.",
)
data_option = click.option(
    "--data", "data_path", type=click.Path(path_type=Path), default=None,
    help="Override the raw CSV location.",
)
verbose_option = click.option("--debug", is_flag=True, help="Enable debug logging.")


@click.group()
def cli() -> None:
    """Appliance energy forecasting pipeline."""


@cli.command()
@config_option
@data_option
@verbose_option
@handle_errors
def validate(config_path: Optional[Path], data_path: Optional[Path], debug: bool) -> None:
    """Load the dataset and print its quality report."""
    configure_logging(logging.DEBUG if debug else logging.INFO)
    from energy_forecast.data import build_dataset

    _, report = build_dataset(load_config(config_path), data_path)
    for key, value in report.to_dict().items():
        click.echo(f"{key:24s} {value}")


@cli.command()
@config_option
@data_option
@verbose_option
@handle_errors
def audit(config_path: Optional[Path], data_path: Optional[Path], debug: bool) -> None:
    """Print the serving-time availability audit for every engineered feature."""
    configure_logging(logging.DEBUG if debug else logging.INFO)
    from energy_forecast.data import build_dataset
    from energy_forecast.features import FeatureBuilder

    config = load_config(config_path)
    frame, _ = build_dataset(config, data_path)
    builder = FeatureBuilder(config)
    builder.build(frame)
    table = builder.audit()

    click.echo(table.groupby(["family", "availability"]).size().to_string())
    click.echo(f"\ntotal features: {len(table)}")
    click.echo(f"minimum lag among observed features: "
               f"{table.loc[table.availability == 'lagged_observation', 'min_lag_steps'].min()}")


@cli.command()
@config_option
@data_option
@verbose_option
@click.option("--skip-deep", is_flag=True, help="Baselines only; does not require TensorFlow.")
@click.option("--keras-verbose", type=int, default=0, help="Keras fit verbosity.")
@handle_errors
def train(config_path: Optional[Path], data_path: Optional[Path], debug: bool,
          skip_deep: bool, keras_verbose: int) -> None:
    """Run the pipeline end to end and write metrics and figures."""
    config = load_config(config_path)
    reports_dir = config.output_dir("reports_dir")
    configure_logging(logging.DEBUG if debug else logging.INFO, reports_dir / "run.log")

    from energy_forecast import evaluate as ev
    from energy_forecast.training.pipeline import run_all

    ev.use_headless_backend()
    artifacts = run_all(config, data_path, skip_deep=skip_deep, verbose=keras_verbose)

    figures_dir = config.output_dir("figures_dir")
    actual = artifacts.test_actual
    index = (artifacts.sequences.test_index if artifacts.sequences is not None
             else artifacts.split.X_test.index)

    ev.save_figure(ev.plot_actual_vs_predicted(index, actual, artifacts.predictions),
                   figures_dir / "actual_vs_predicted.png")
    ev.save_figure(ev.plot_metric_comparison(artifacts.comparison),
                   figures_dir / "metric_comparison.png")

    best = artifacts.comparison.index[0]
    ev.save_figure(
        ev.plot_residual_diagnostics(actual, artifacts.predictions[best][-len(actual):], best),
        figures_dir / "residual_diagnostics.png")
    if artifacts.histories:
        ev.save_figure(ev.plot_training_history(artifacts.histories),
                       figures_dir / "training_history.png")

    ev.write_report(artifacts.metrics, actual, artifacts.predictions,
                    config.path(config.outputs["metrics_file"]),
                    extra={"data_quality": artifacts.quality.to_dict(),
                           "selected_features": artifacts.selection.selected,
                           "trials": [t.to_dict() for t in artifacts.trials]})

    click.echo("\n" + artifacts.comparison.round(3).to_string())
    click.echo("\n" + ev.summarise(artifacts.metrics, actual))


@cli.command()
@config_option
@verbose_option
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path),
              help="Raw CSV to score, in the same schema as the training data.")
@click.option("--output", "output_path", type=click.Path(path_type=Path), default=None,
              help="Where to write the predictions. Printed to stdout when omitted.")
@handle_errors
def predict(config_path: Optional[Path], debug: bool, input_path: Path,
            output_path: Optional[Path]) -> None:
    """Score new data using the saved model and preprocessor."""
    configure_logging(logging.DEBUG if debug else logging.INFO)
    from energy_forecast.prediction.service import ForecastService

    service = ForecastService.load(load_config(config_path))
    predictions = service.predict_csv(input_path, output_path)

    if output_path is None:
        click.echo(predictions.round(1).to_string())
    click.echo(f"\n{len(predictions)} timestamps scored | "
               f"mean {predictions.mean():.1f} Wh | max {predictions.max():.1f} Wh")


@cli.command()
@config_option
@click.option("--metric", default="mae", help="Metric to rank runs by.")
@handle_errors
def runs(config_path: Optional[Path], metric: str) -> None:
    """List every recorded run, best first."""
    from energy_forecast.utils.tracking import compare_runs

    config = load_config(config_path)
    base = config.outputs.get("experiments_dir") or config.outputs["reports_dir"]
    table = compare_runs(config.path(base), metric)
    if table.empty:
        click.echo("no runs recorded yet; run `energy-forecast train` first")
    else:
        click.echo(table.to_string())


if __name__ == "__main__":  # pragma: no cover
    cli()

"""Metrics and figures. Every metric is reported in the original units (Wh)."""

from energy_forecast.evaluation.metrics import (
    Metrics,
    compute_metrics,
    error_by_level,
    metrics_table,
)
from energy_forecast.evaluation.plots import (
    plot_actual_vs_predicted,
    plot_metric_comparison,
    plot_residual_diagnostics,
    plot_training_history,
    save_figure,
    summarise,
    use_headless_backend,
    write_report,
)

__all__ = [
    "Metrics", "compute_metrics", "error_by_level", "metrics_table",
    "plot_actual_vs_predicted", "plot_metric_comparison", "plot_residual_diagnostics",
    "plot_training_history", "save_figure", "summarise", "use_headless_backend",
    "write_report",
]

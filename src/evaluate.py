"""Compute regression metrics in Wh and plot model diagnostics."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 170, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
GREEN = "#177c62"


def regression_metrics(actual, predicted):
    """MAE/RMSE have Wh units; MAPE is percent over nonzero labels only."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    nonzero = actual != 0
    return {"MAE_Wh": float(mean_absolute_error(actual, predicted)),
            "RMSE_Wh": float(np.sqrt(mean_squared_error(actual, predicted))),
            "R2": float(r2_score(actual, predicted)),
            "MAPE_pct": float(np.mean(np.abs((actual[nonzero]-predicted[nonzero])/actual[nonzero]))*100)}


def save_figure(fig, path):
    """Apply a compact layout and close figures to avoid notebook memory growth."""
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_eda(frame, train_end, test_start, acf, importances, outdir):
    """Full timeline for context; all pattern and feature-choice plots use training only."""
    outdir.mkdir(parents=True, exist_ok=True)
    train = frame.iloc[:train_end]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    frame.Appliances.resample("D").mean().plot(ax=ax, color=GREEN)
    ax.axvline(frame.index[train_end], color="#bd7c00", label="Validation begins")
    ax.axvline(frame.index[test_start], color="#b34444", label="Test begins")
    ax.set(title="Daily mean appliance energy (partial boundary days included)", ylabel="Wh per 10-minute interval", xlabel="Date")
    ax.legend(fontsize=8)
    save_figure(fig, outdir / "01_timeline.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    sns.histplot(train.Appliances, bins=50, ax=axes[0], color=GREEN)
    axes[0].set(title="Training target distribution", xlabel="Appliance energy (Wh)")
    sns.boxplot(x=train.index.hour, y=train.Appliances.to_numpy(), ax=axes[1], color="#81bfad", fliersize=1)
    axes[1].set(title="Training energy by hour", xlabel="Hour", ylabel="Wh")
    axes[1].tick_params(axis="x", labelsize=7)
    save_figure(fig, outdir / "02_distribution_hour.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    train.Appliances.groupby(train.index.dayofweek).mean().plot.bar(ax=axes[0], color=GREEN)
    axes[0].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], rotation=0)
    axes[0].set(title="Training weekday pattern", ylabel="Mean Wh", xlabel="Day")
    axes[1].plot(acf.index, acf.values, color=GREEN)
    axes[1].set(title="Training autocorrelation", xlabel="Lag (10-minute steps)", ylabel="Pearson correlation")
    for lag in [6, 18, 144]:
        axes[1].axvline(lag, color="grey", lw=.5, alpha=.5)
    save_figure(fig, outdir / "03_weekday_acf.png")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(train.corr(), cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": .6})
    ax.set_title("Training correlations: descriptive, same-timestamp associations")
    ax.tick_params(labelsize=7)
    save_figure(fig, outdir / "04_correlations.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    importances.sort_values().tail(20).plot.barh(ax=ax, color=GREEN)
    ax.set(title="Training ExtraTrees importance: top 20", xlabel="Impurity-based importance (not causality)")
    save_figure(fig, outdir / "05_feature_importance.png")


def make_results_plots(predictions, metrics, histories, selected_name, outdir):
    """Visualize preselected model errors and all locked test comparisons."""
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["MAE_Wh", "RMSE_Wh"]):
        metrics.set_index("model")[metric].plot.barh(ax=ax, color=GREEN)
        ax.set(xlabel=metric.replace("_", " "), ylabel="", title=f"Test {metric.split('_')[0]}")
    save_figure(fig, outdir / "06_metric_comparison.png")
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    for ax, subset, title in [(axes[0], predictions, "Entire test period"),
                             (axes[1], predictions.iloc[:288], "First 48 test hours")]:
        ax.plot(subset.index, subset.actual, lw=.7, label="Actual", color="#404040")
        ax.plot(subset.index, subset[selected_name], lw=.7, alpha=.85, label=selected_name, color=GREEN)
        ax.set(title=title, ylabel="Wh")
        locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.legend(fontsize=8)
    save_figure(fig, outdir / "07_predictions.png")
    residual = predictions.actual - predictions[selected_name]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    axes[0].scatter(predictions.actual, predictions[selected_name], s=4, alpha=.25, color=GREEN)
    lim = max(predictions.actual.max(), predictions[selected_name].max())
    axes[0].plot([0, lim], [0, lim], "k--", lw=1)
    axes[0].set(xlabel="Actual Wh", ylabel="Predicted Wh", title="Predicted versus actual")
    sns.histplot(residual, bins=45, ax=axes[1], color=GREEN)
    axes[1].set(xlabel="Actual - predicted (Wh)", title="Residual distribution")
    axes[2].scatter(predictions[selected_name], residual, s=4, alpha=.25, color=GREEN)
    axes[2].axhline(0, color="black", ls="--", lw=1)
    axes[2].set(xlabel="Predicted Wh", ylabel="Residual Wh", title="Residuals versus prediction")
    save_figure(fig, outdir / "08_residuals.png")
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, history in histories.items():
        h = pd.DataFrame(history)
        line, = ax.plot(h.epoch, h.val_mse, label=name)
        ax.plot(h.epoch, h.train_mse, ls="--", alpha=.65, color=line.get_color())
    ax.set(title="Learning curves: solid validation, dashed training", xlabel="Epoch", ylabel="Standardized-target MSE")
    ax.legend(fontsize=7, ncol=2)
    save_figure(fig, outdir / "09_learning_curves.png")

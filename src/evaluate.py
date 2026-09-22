"""Model evaluation metrics, alignment utilities, and diagnostic summaries."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred, label=""):
    """Compute MAE, RMSE, MAPE (over non-zero values), and R2 in physical units (Wh)."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    nz = y_true != 0

    return {
        "Model": label,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_%": float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100),
        "R2": float(r2_score(y_true, y_pred)),
    }


def aligned(pred, target_length):
    """Trim a baseline prediction to the rows that sequence models cover."""
    pred_arr = np.asarray(pred, float)
    if len(pred_arr) < target_length:
        raise ValueError(f"Prediction array length ({len(pred_arr)}) is shorter than target ({target_length})")
    return pred_arr[-target_length:]


def evaluate_consumption_slices(y_true, y_pred, bins=(0, 50, 100, 200, 400, np.inf),
                                labels=("<50", "50-100", "100-200", "200-400", ">400")):
    """Break down error metrics across consumption magnitude strata."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    abs_errors = np.abs(y_true - y_pred)

    levels = pd.cut(y_true, bins=bins, labels=labels)
    df_slices = pd.DataFrame({"error": abs_errors, "level": levels})
    return df_slices.groupby("level", observed=False).agg(
        mean_abs_error=("error", "mean"),
        samples=("error", "count")
    ).round(2)

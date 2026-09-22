"""Reload saved preprocessing and selected deep model to predict one NEW reading."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from .data_preprocessing import load_data
from .feature_engineering import make_features
from .model import EnergyRNN


def forecast_next(history_path, model_root="."):
    """Forecast the timestamp 10 minutes after the last observed history record.

    Supply the SAME sensor schema and enough recent measured energy history.
    This is one-step forecasting, not a recursive day-ahead forecasting API.
    Only load checkpoints/joblib files produced by this trusted project.
    """
    root = Path(model_root)
    selected = json.loads((root/"reports"/"selection_frozen_before_test.json").read_text())["selected_deep_model"]
    preprocessing = joblib.load(root/"models"/"preprocessing.joblib")
    checkpoint = torch.load(root/"models"/f"{selected}.pt", map_location="cpu", weights_only=True)
    trial = checkpoint["trial"]
    frame = load_data(history_path)
    required = preprocessing["max_lag"] + trial["lookback"] - 1
    if len(frame) < required:
        raise ValueError(f"Need at least {required} consecutive measured history rows.")
    next_date = frame.index[-1] + pd.Timedelta(minutes=10)
    future_row = pd.DataFrame(np.nan, index=pd.DatetimeIndex([next_date], name="date"), columns=frame.columns)
    extended = pd.concat([frame, future_row])
    features = make_features(extended, preprocessing["lags"])[preprocessing["all_columns"]]
    scaled = preprocessing["x_scaler"].transform(preprocessing["imputer"].transform(features))
    positions = [preprocessing["all_columns"].index(name) for name in preprocessing["selected"]]
    sequence = scaled[-trial["lookback"]:, positions].astype(np.float32)
    model = EnergyRNN(checkpoint["n_features"], **{key: trial[key] for key in ["kind", "hidden", "layers", "dropout"]})
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        standardized = model(torch.from_numpy(sequence[None])).item()
    energy = preprocessing["y_scaler"].inverse_transform([[standardized]])[0, 0]
    return {"forecast_timestamp": str(next_date), "predicted_energy_Wh": float(max(0, energy)),
            "model": selected, "horizon_minutes": 10,
            "note": "Point forecast; actual next reading is not supplied or known."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", default="data/raw/energy_data_set.csv")
    parser.add_argument("--model-root", default=".")
    args = parser.parse_args()
    print(json.dumps(forecast_next(args.history, args.model_root), indent=2))

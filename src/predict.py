"""Inference and operational forecasting for the next 10-minute interval."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from .data_preprocessing import load_data, winsorise
from .feature_engineering import build_features


def forecast_next(history_data, model_path="models/best_model.keras", preprocessor_path="models/preprocessor.joblib"):
    """Forecast appliance energy consumption (Wh) for the next 10-minute interval.

    Parameters
    ----------
    history_data : str, Path, or pd.DataFrame
        Path to recent history CSV or a DataFrame of past readings.
        Must contain at least 1 week + lookback of readings to construct features without NaN.
    model_path : str or Path
        Path to the saved TensorFlow/Keras model (.keras).
    preprocessor_path : str or Path
        Path to the saved preprocessor dictionary (.joblib).

    Returns
    -------
    dict
        Forecast metadata including timestamp, predicted Wh, and model info.
    """
    model_p = Path(model_path)
    prep_p = Path(preprocessor_path)

    if not model_p.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run training first.")
    if not prep_p.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found at {preprocessor_path}.")

    preprocessor = joblib.load(prep_p)
    model = keras.models.load_model(model_p)

    if isinstance(history_data, (str, Path)):
        df = load_data(history_data)
    else:
        df = history_data.copy()

    lookback = preprocessor["lookback"]
    target_col = preprocessor["target"]
    selected_features = preprocessor["selected_features"]
    fences = preprocessor["fences"]
    x_scaler = preprocessor["x_scaler"]
    y_scaler = preprocessor["y_scaler"]

    # Build features on history
    X_features, _ = build_features(df, target=target_col)

    if len(X_features) < lookback:
        raise ValueError(f"Need at least {lookback} feature rows after warm-up; received {len(X_features)}.")

    # Winsorise and scale
    X_winsor = winsorise(X_features, fences)
    X_scaled = x_scaler.transform(X_winsor)
    X_df_scaled = pd.DataFrame(X_scaled, index=X_features.index, columns=X_features.columns)

    # Extract sequence window for prediction
    recent_window = X_df_scaled[selected_features].iloc[-lookback:].values
    input_sequence = np.expand_dims(recent_window, axis=0)  # Shape: (1, lookback, n_features)

    # Predict in model space
    pred_scaled = model.predict(input_sequence, verbose=0).ravel()[0]

    # Invert to physical Wh units
    pred_wh = float(y_scaler.inverse_transform([pred_scaled])[0])

    last_timestamp = df.index[-1]
    forecast_timestamp = last_timestamp + pd.Timedelta(minutes=10)

    return {
        "last_observed_timestamp": str(last_timestamp),
        "forecast_timestamp": str(forecast_timestamp),
        "forecast_horizon": "10 minutes",
        "predicted_energy_Wh": round(pred_wh, 2),
        "model_file": str(model_p.name),
        "features_used": len(selected_features),
    }


def main():
    parser = argparse.ArgumentParser(description="Predict next 10-minute appliance energy consumption.")
    parser.add_argument("--history", default="data/raw/energy_data_set.csv", help="Path to history CSV.")
    parser.add_argument("--model", default="models/best_model.keras", help="Path to trained .keras model.")
    parser.add_argument("--preprocessor", default="models/preprocessor.joblib", help="Path to preprocessor .joblib.")
    args = parser.parse_args()

    result = forecast_next(args.history, model_path=args.model, preprocessor_path=args.preprocessor)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

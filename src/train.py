"""End-to-end training pipeline for appliance energy forecasting."""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

from .data_preprocessing import (
    load_data, enforce_regular_grid, calculate_winsor_bounds, winsorise,
    TargetScaler, save_preprocessed_data, TARGET
)
from .feature_engineering import build_features, select_features
from .model import (
    make_sequences, build_lstm, build_gru, build_cnn_lstm,
    build_tuned, train_model, DEFAULT_LOOKBACK, BUILDERS
)
from .evaluate import regression_metrics, aligned

RANDOM_STATE = 42


def set_seed(seed=RANDOM_STATE):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def run_pipeline(data_path, output_dir=".", lookback=DEFAULT_LOOKBACK, n_trials=8):
    """Execute the full training, tuning, and evaluation pipeline."""
    set_seed(RANDOM_STATE)
    start_time = time.perf_counter()

    root = Path(output_dir)
    models_dir = root / "models"
    reports_dir = root / "reports"
    processed_dir = root / "data" / "processed"

    for d in [models_dir, reports_dir, processed_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"1. Loading dataset from {data_path}...")
    df = load_data(data_path)
    df = enforce_regular_grid(df)

    print("2. Engineering leakage-safe temporal and domain features...")
    X_all, y_all = build_features(df, target=TARGET)
    n = len(X_all)

    # Chronological 72% / 8% / 20% split
    i_test = int(n * 0.80)
    i_val = int(i_test * 0.90)

    X_train, y_train = X_all.iloc[:i_val], y_all.iloc[:i_val]
    X_val, y_val = X_all.iloc[i_val:i_test], y_all.iloc[i_val:i_test]
    X_test, y_test = X_all.iloc[i_test:], y_all.iloc[i_test:]

    print(f"   Train: {len(X_train)} rows | Val: {len(X_val)} rows | Test: {len(X_test)} rows")

    print("3. Winsorising and scaling...")
    bounds = calculate_winsor_bounds(X_train, k=3.0)
    X_train_w = winsorise(X_train, bounds)
    X_val_w = winsorise(X_val, bounds)
    X_test_w = winsorise(X_test, bounds)

    x_scaler = StandardScaler().fit(X_train_w)
    Xtr_s = x_scaler.transform(X_train_w)
    Xva_s = x_scaler.transform(X_val_w)
    Xte_s = x_scaler.transform(X_test_w)

    y_scaler = TargetScaler().fit(y_train)
    ytr_s = y_scaler.transform(y_train)
    yva_s = y_scaler.transform(y_val)
    yte_s = y_scaler.transform(y_test)

    print("4. Selecting top features via consensus voting...")
    selected_features, rf_imp, corr_rank = select_features(
        pd.DataFrame(Xtr_s, columns=X_all.columns),
        ytr_s,
        n_top=30,
        random_state=RANDOM_STATE
    )
    sel_idx = [X_all.columns.get_loc(c) for c in selected_features]
    Xtr_sel = Xtr_s[:, sel_idx]
    Xva_sel = Xva_s[:, sel_idx]
    Xte_sel = Xte_s[:, sel_idx]

    # Save preprocessor artifact
    preprocessor_payload = {
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "selected_features": selected_features,
        "all_feature_names": list(X_all.columns),
        "fences": bounds,
        "lookback": lookback,
        "target": TARGET,
    }
    joblib.dump(preprocessor_payload, models_dir / "preprocessor.joblib")
    save_preprocessed_data(Xtr_sel, y_train, Xva_sel, y_val, Xte_sel, y_test, processed_dir)

    results = []
    predictions = {}

    print("5. Training baseline models...")
    # Baseline 1: Persistence
    pred_persist = X_test["app_lag1"].values
    results.append(regression_metrics(y_test.values, pred_persist, "Persistence (y[t-1])"))
    predictions["Persistence (y[t-1])"] = pred_persist

    # Baseline 2: Ridge
    ridge = Ridge(alpha=1.0).fit(Xtr_sel, ytr_s)
    pred_ridge = y_scaler.inverse_transform(ridge.predict(Xte_sel))
    results.append(regression_metrics(y_test.values, pred_ridge, "Ridge regression"))
    predictions["Ridge regression"] = pred_ridge
    joblib.dump(ridge, models_dir / "ridge_model.joblib")

    # Baseline 3: Random Forest
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=20, min_samples_leaf=2,
        n_jobs=-1, random_state=RANDOM_STATE
    ).fit(Xtr_sel, ytr_s)
    pred_rf = y_scaler.inverse_transform(rf.predict(Xte_sel))
    results.append(regression_metrics(y_test.values, pred_rf, "Random forest"))
    predictions["Random forest"] = pred_rf
    joblib.dump(rf, models_dir / "random_forest_model.joblib", compress=3)

    print("6. Preparing 3-D sliding window sequences...")
    X_full_sel = np.vstack([Xtr_sel, Xva_sel, Xte_sel])
    y_full_s = np.concatenate([ytr_s, yva_s, yte_s])

    X_seq, y_seq = make_sequences(X_full_sel, y_full_s, lookback=lookback)
    seq_target_pos = np.arange(lookback, len(y_full_s))

    train_mask = seq_target_pos < i_val
    val_mask = (seq_target_pos >= i_val) & (seq_target_pos < i_test)
    test_mask = seq_target_pos >= i_test

    Xtr_seq, ytr_seq = X_seq[train_mask], y_seq[train_mask]
    Xva_seq, yva_seq = X_seq[val_mask], y_seq[val_mask]
    Xte_seq, yte_seq = X_seq[test_mask], y_seq[test_mask]

    y_test_seq_wh = y_scaler.inverse_transform(yte_seq)
    test_index = X_all.index[seq_target_pos[test_mask]]
    input_shape = (lookback, Xtr_seq.shape[2])

    print("7. Training deep recurrent architectures (LSTM, GRU, CNN-LSTM)...")
    dl_models = {
        "LSTM": build_lstm(input_shape),
        "GRU": build_gru(input_shape),
        "CNN-LSTM": build_cnn_lstm(input_shape)
    }

    trained_models = {}
    for name, model in dl_models.items():
        print(f"   Training {name}...")
        set_seed(RANDOM_STATE)
        keras.backend.clear_session()
        train_model(model, Xtr_seq, ytr_seq, Xva_seq, yva_seq, epochs=40, batch_size=64, patience=6)
        trained_models[name] = model

        pred = y_scaler.inverse_transform(model.predict(Xte_seq, verbose=0).ravel())
        predictions[name] = pred
        row = regression_metrics(y_test_seq_wh, pred, name)
        results.append(row)
        clean_name = name.lower().replace("-", "_")
        model.save(models_dir / f"{clean_name}_model.keras")

    # Determine best architecture
    comparison = pd.DataFrame(results).set_index("Model")
    recurrent_names = [m for m in comparison.index if m in {"LSTM", "GRU", "CNN-LSTM"}]
    best_arch = comparison.loc[recurrent_names, "MAE"].idxmin()
    print(f"8. Hyperparameter optimization on best architecture: {best_arch}...")

    search_space = {
        "units": [32, 64, 96],
        "dropout": [0.1, 0.2, 0.3],
        "lr": [3e-4, 1e-3, 3e-3],
        "batch_size": [32, 64, 128],
        "optimizer": ["adam", "rmsprop"],
    }
    rng = np.random.default_rng(RANDOM_STATE)
    yva_wh = y_scaler.inverse_transform(yva_seq)
    trials = []

    for trial in range(min(n_trials, 8)):
        cfg = {k: v[int(rng.integers(len(v)))] for k, v in search_space.items()}
        set_seed(RANDOM_STATE)
        keras.backend.clear_session()
        trial_model = build_tuned(best_arch, input_shape, int(cfg["units"]), float(cfg["dropout"]), float(cfg["lr"]), cfg["optimizer"])
        train_model(trial_model, Xtr_seq, ytr_seq, Xva_seq, yva_seq, epochs=30, batch_size=int(cfg["batch_size"]), patience=5)
        val_mae = float(np.mean(np.abs(yva_wh - y_scaler.inverse_transform(trial_model.predict(Xva_seq, verbose=0).ravel()))))
        trials.append({**cfg, "val_MAE_Wh": val_mae})

    best_cfg = pd.DataFrame(trials).sort_values("val_MAE_Wh").iloc[0].to_dict()
    print(f"   Best trial configuration: {best_cfg}")

    set_seed(RANDOM_STATE)
    keras.backend.clear_session()
    tuned_model = build_tuned(best_arch, input_shape, int(best_cfg["units"]), float(best_cfg["dropout"]), float(best_cfg["lr"]), best_cfg["optimizer"])
    train_model(tuned_model, Xtr_seq, ytr_seq, Xva_seq, yva_seq, epochs=60, batch_size=int(best_cfg["batch_size"]), patience=8)

    tuned_label = f"{best_arch} (tuned)"
    pred_tuned = y_scaler.inverse_transform(tuned_model.predict(Xte_seq, verbose=0).ravel())
    predictions[tuned_label] = pred_tuned
    results.append(regression_metrics(y_test_seq_wh, pred_tuned, tuned_label))

    # Save models
    tuned_model.save(models_dir / "tuned_model.keras")
    tuned_model.save(models_dir / "best_model.keras")

    print("9. Generating and saving reports...")
    final_table = (
        pd.DataFrame(results).drop_duplicates(subset="Model", keep="last")
        .set_index("Model").sort_values("MAE").round(3)
    )
    final_table.to_csv(reports_dir / "final_model_comparison.csv")

    pred_df = pd.DataFrame({"actual_Wh": y_test_seq_wh}, index=test_index)
    for name, p in predictions.items():
        pred_df[name] = aligned(p, len(y_test_seq_wh))
    pred_df.to_csv(reports_dir / "test_predictions.csv")

    summary = {
        "best_model": final_table.index[0],
        "best_MAE_Wh": float(final_table.iloc[0]["MAE"]),
        "best_RMSE_Wh": float(final_table.iloc[0]["RMSE"]),
        "persistence_MAE_Wh": float(final_table.loc["Persistence (y[t-1])", "MAE"]),
        "total_runtime_seconds": float(time.perf_counter() - start_time),
    }
    with open(reports_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- FINAL TEST RESULTS ---")
    print(final_table)
    print(f"\nTraining pipeline completed in {summary['total_runtime_seconds']:.1f}s.")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Appliance energy forecasting training pipeline.")
    parser.add_argument("--data", default="data/raw/energy_data_set.csv", help="Path to raw CSV.")
    parser.add_argument("--output", default=".", help="Output directory for models and reports.")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK, help="Sequence history window length.")
    parser.add_argument("--trials", type=int, default=6, help="Number of tuning trials.")
    args = parser.parse_args()

    run_pipeline(args.data, output_dir=args.output, lookback=args.lookback, n_trials=args.trials)


if __name__ == "__main__":
    main()

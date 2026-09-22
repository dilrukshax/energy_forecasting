"""Train all candidates, freeze the validation decision, then evaluate once on test.

Run from the project root: python -m src.train
"""
import argparse
import copy
import importlib.metadata
import json
import platform
import random
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from .data_preprocessing import load_data, split_boundaries, audit_data, fit_feature_preprocessing
from .feature_engineering import choose_lags, make_features, select_features, MAX_LAG, MAX_LOOKBACK, WARMUP
from .model import EnergyRNN, SequenceDataset
from .evaluate import regression_metrics, make_eda, make_results_plots


def write_json(path, payload):
    """Write a readable UTF-8 artifact; no NaN/Infinity JSON values are permitted."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def set_seed(seed, threads):
    """Control Python/NumPy/PyTorch randomness; deterministic CPU is the default."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def predict_scaled(model, loader):
    """Inference disables dropout and gradient recording."""
    model.eval()
    results = []
    with torch.no_grad():
        for x_batch, _ in loader:
            results.append(model(x_batch).cpu().numpy())
    return np.concatenate(results)


def fit_recurrent(x, y_scaled, train_indices, val_indices, trial, settings):
    """Fit with AdamW, gradient clipping, dropout and best-validation checkpointing."""
    set_seed(settings["seed"], settings["threads"])
    model = EnergyRNN(x.shape[1], **{key: trial[key] for key in ["kind", "hidden", "layers", "dropout"]})
    # No sequence or time split is randomized; chronological batches are explicit.
    train_loader = DataLoader(SequenceDataset(x, y_scaled, train_indices, trial["lookback"]),
                              batch_size=trial["batch_size"], shuffle=False)
    val_loader = DataLoader(SequenceDataset(x, y_scaled, val_indices, trial["lookback"]), batch_size=256, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=trial["lr"], weight_decay=1e-4)
    criterion = nn.MSELoss()
    best_loss, stale, best_epoch, best_state = float("inf"), 0, 0, None
    history = []
    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
        predicted = predict_scaled(model, val_loader)
        val_loss = float(np.mean((predicted - y_scaled[val_indices])**2))
        history.append({"epoch": epoch, "train_mse": total_loss/len(train_indices), "val_mse": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss, stale, best_epoch = val_loss, 0, epoch
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        print(f"{trial['name']} epoch={epoch:02d} train_mse={history[-1]['train_mse']:.4f} val_mse={val_loss:.4f}", flush=True)
        if stale >= settings["patience"]:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a finite validation checkpoint.")
    model.load_state_dict(best_state)
    return model, history, best_epoch


def run_pipeline(data_path, config_path, output_root, overwrite=False):
    """Execute a fixed experiment. Test labels never choose features or hyperparameters."""
    root = Path(output_root)
    reports, models = root/"reports", root/"models"
    for directory in [reports, models]:
        directory.mkdir(parents=True, exist_ok=True)
    if (reports/"run_summary.json").exists() and not overwrite:
        raise FileExistsError("Completed results exist. Use another --output directory, or --overwrite to reproduce the fixed run.")
    settings = json.loads(Path(config_path).read_text())
    if any(t["lookback"] > MAX_LOOKBACK for t in settings["trials"]):
        raise ValueError(f"Lookbacks above {MAX_LOOKBACK} require a new common warmup protocol.")
    if settings["max_epochs"] < 1 or settings["patience"] < 1:
        raise ValueError("max_epochs and patience must be positive.")
    set_seed(settings["seed"], settings["threads"])
    started = time.perf_counter()
    frame = load_data(data_path)
    train_end, test_start = split_boundaries(len(frame))
    if train_end <= WARMUP:
        raise ValueError("Not enough training data after history warmup.")
    audit = audit_data(frame, data_path)
    write_json(reports/"data_audit.json", audit)
    lags, acf = choose_lags(frame.Appliances.iloc[:train_end])
    features = make_features(frame, lags)
    all_columns = features.columns.tolist()
    all_x, imputer, x_scaler = fit_feature_preprocessing(features, train_end, MAX_LAG)
    train_indices = np.arange(WARMUP, train_end)
    val_indices = np.arange(train_end, test_start)
    test_indices = np.arange(test_start, len(frame))
    y = frame.Appliances.to_numpy(dtype=np.float32)
    y_scaler = StandardScaler().fit(y[train_indices, None])
    y_scaled = y_scaler.transform(y[:, None]).ravel().astype(np.float32)
    # Feature ranking uses training labels only, not validation or test labels.
    ranking_model = ExtraTreesRegressor(n_estimators=128, min_samples_leaf=5, random_state=settings["seed"], n_jobs=settings["threads"])
    ranking_model.fit(all_x[train_indices], y[train_indices])
    importances = pd.Series(ranking_model.feature_importances_, index=all_columns, name="importance")
    selected = select_features(importances, settings["top_k"])
    selected_positions = [all_columns.index(name) for name in selected]
    x = all_x[:, selected_positions]
    importances.sort_values(ascending=False).to_csv(reports/"feature_importance.csv", index_label="feature")
    acf.to_csv(reports/"training_acf.csv", index_label="lag_steps")
    write_json(reports/"selected_features.json", {"lags": lags, "selected": selected, "all": all_columns})
    split_table = pd.DataFrame([
        {"split": name, "first_target": str(frame.index[indices[0]]), "last_target": str(frame.index[indices[-1]]), "samples": len(indices)}
        for name, indices in [("train", train_indices), ("validation", val_indices), ("test", test_indices)]
    ])
    split_table.to_csv(reports/"splits.csv", index=False)
    preprocessing = {"imputer": imputer, "x_scaler": x_scaler, "y_scaler": y_scaler,
                     "all_columns": all_columns, "selected": selected, "lags": lags,
                     "max_lag": MAX_LAG, "forecast_horizon_minutes": 10}
    joblib.dump(preprocessing, models/"preprocessing.joblib", compress=3)
    make_eda(frame, train_end, test_start, acf, importances, reports/"figures")
    val_predictions = {"Persistence": frame.Appliances.shift(1).iloc[val_indices].to_numpy(),
                       "Daily_seasonal": frame.Appliances.shift(144).iloc[val_indices].to_numpy()}
    baselines = {
        "Ridge": (Ridge(alpha=10.0), x),
        "RandomForest_selected": (RandomForestRegressor(n_estimators=160, min_samples_leaf=5,
                                  max_features=.8, random_state=settings["seed"], n_jobs=settings["threads"]), x),
        "RandomForest_all": (RandomForestRegressor(n_estimators=160, min_samples_leaf=5,
                             max_features=.8, random_state=settings["seed"], n_jobs=settings["threads"]), all_x)
    }
    for name, (model, matrix) in baselines.items():
        model.fit(matrix[train_indices], y[train_indices])
        val_predictions[name] = np.maximum(model.predict(matrix[val_indices]), 0)
        joblib.dump(model, models/f"{name}.joblib", compress=3)
        print(f"Fitted {name}", flush=True)
    recurrent, histories, tuning_rows = {}, {}, []
    for trial in settings["trials"]:
        trial_start = time.perf_counter()
        model, history, epoch = fit_recurrent(x, y_scaled, train_indices, val_indices, trial, settings)
        name = trial["name"]
        loader = DataLoader(SequenceDataset(x, y_scaled, val_indices, trial["lookback"]), batch_size=256, shuffle=False)
        predicted = y_scaler.inverse_transform(predict_scaled(model, loader)[:, None]).ravel()
        val_predictions[name] = np.maximum(predicted, 0)
        recurrent[name] = (model, trial)
        histories[name] = history
        torch.save({"state_dict": model.state_dict(), "trial": trial, "n_features": x.shape[1]}, models/f"{name}.pt")
        tuning_rows.append({**trial, "best_epoch": epoch, "epochs_run": len(history),
                            "parameters": sum(p.numel() for p in model.parameters()),
                            "seconds": time.perf_counter()-trial_start,
                            **regression_metrics(y[val_indices], val_predictions[name])})
        pd.DataFrame(tuning_rows).to_csv(reports/"tuning_results.csv", index=False)
        write_json(reports/"learning_history.json", histories)
    validation = pd.DataFrame([{"model": name, **regression_metrics(y[val_indices], p)} for name, p in val_predictions.items()])
    validation.to_csv(reports/"validation_metrics.csv", index=False)
    selected_name = validation[validation.model.isin(recurrent)].sort_values("RMSE_Wh").iloc[0].model
    champion = validation.sort_values("RMSE_Wh").iloc[0].model
    # This file is written BEFORE test predictions: the decision cannot depend on test RMSE.
    selection = {"selection_metric": "validation RMSE in Wh after nonnegative clipping",
                 "selected_deep_model": selected_name, "overall_validation_champion": champion,
                 "baseline_reference": "LSTM_initial", "test_used_for_selection": False,
                 "refit_on_validation": False}
    write_json(reports/"selection_frozen_before_test.json", selection)
    test_predictions = {"Persistence": frame.Appliances.shift(1).iloc[test_indices].to_numpy(),
                        "Daily_seasonal": frame.Appliances.shift(144).iloc[test_indices].to_numpy()}
    for name, (model, matrix) in baselines.items():
        test_predictions[name] = np.maximum(model.predict(matrix[test_indices]), 0)
    for name, (model, trial) in recurrent.items():
        loader = DataLoader(SequenceDataset(x, y_scaled, test_indices, trial["lookback"]), batch_size=256, shuffle=False)
        p = y_scaler.inverse_transform(predict_scaled(model, loader)[:, None]).ravel()
        test_predictions[name] = np.maximum(p, 0)
    metrics = pd.DataFrame([{"model": name, **regression_metrics(y[test_indices], p)} for name, p in test_predictions.items()])
    metrics.to_csv(reports/"test_metrics.csv", index=False)
    predictions = pd.DataFrame({"actual": y[test_indices], **test_predictions}, index=frame.index[test_indices])
    predictions.to_csv(reports/"test_predictions.csv", index_label="date")
    high_threshold = float(np.quantile(y[train_indices], .95))
    high_mask = y[test_indices] >= high_threshold
    slice_metrics = []
    for label, mask in [("below_training_p95", ~high_mask), ("at_or_above_training_p95", high_mask)]:
        if mask.any():
            slice_metrics.append({"slice": label, "threshold_Wh": high_threshold, "samples": int(mask.sum()),
                                  **regression_metrics(y[test_indices][mask], test_predictions[selected_name][mask])})
    pd.DataFrame(slice_metrics).to_csv(reports/"peak_error_analysis.csv", index=False)
    make_results_plots(predictions, metrics, histories, selected_name, reports/"figures")
    versions = {pkg: importlib.metadata.version(pkg) for pkg in ["numpy", "pandas", "scikit-learn", "torch", "matplotlib", "seaborn", "joblib"]}
    summary = {**selection, "settings": settings, "device": "cpu", "python": platform.python_version(),
               "versions": versions, "seed": settings["seed"], "n_features_before_selection": len(all_columns),
               "n_features_selected": len(selected), "history_warmup_rows": WARMUP,
               "train_samples": len(train_indices), "validation_samples": len(val_indices), "test_samples": len(test_indices),
               "lags": lags, "total_seconds": time.perf_counter()-started,
               "test_protocol": "Rolling one-step: earlier true observations become available before later forecasts; no weight updates on test.",
               "source_sha256": audit["source_sha256"]}
    write_json(reports/"run_summary.json", summary)
    print(metrics.to_string(index=False), flush=True)
    print(f"Selected using validation only: {selected_name}; overall champion: {champion}", flush=True)
    return summary


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/energy_data_set.csv")
    parser.add_argument("--config", default="configs/assessment.json")
    parser.add_argument("--output", default=".")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_pipeline(args.data, args.config, args.output, args.overwrite)


if __name__ == "__main__":
    main()

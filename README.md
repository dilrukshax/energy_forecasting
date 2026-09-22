# Appliance Energy Prediction Using Deep Learning

Predict the next 10-minute appliance energy consumption in Watt-hours (Wh) for a low-energy residence using past energy usage, environmental sensors (temperature and humidity across 9 indoor zones plus outdoor), lighting, and calendar features. 

The project compares Deep Recurrent Neural Networks (**LSTM**, **GRU**, and **CNN-LSTM**) against strong regularized benchmarks (**Persistence**, **Ridge Regression**, and **Random Forest**).

The complete end-to-end analysis, exploratory visualizations, methodology, model tuning, and holdout evaluations are available in [`notebooks/appliance_energy_prediction.ipynb`](notebooks/appliance_energy_prediction.ipynb).

---

## Key Features & Methodology

- **Strict Temporal Discipline (No Data Leakage):**
  - Features at time $t$ use only measurements strictly prior to $t$ ($\le t-1$) and deterministic calendar features at $t$.
  - 1-week warm-up ensures all rolling and lag features are strictly computed from observed historical windows.
  - Chronological 72% / 8% / 20% train / validation / test partition.
  - Winsorisation fences and standardisation scalers are derived exclusively from training observations.
- **Rich Feature Engineering:**
  - Lags: 10m, 20m, 30m, 40m, 50m, 1h, 2h, 3h, 6h, 24h, and 7-day target lags.
  - Rolling aggregates: Mean, standard deviation, min, and max over 1h, 3h, 6h, and 24h windows.
  - Momentum & acceleration: 10m, 30m, 1h, 24h differences and 10m acceleration.
  - Calendar & cyclics: Hour, day of week, month, weekend indicator, second-of-day, and sine/cosine cyclical transforms.
  - Domain interactions: Indoor temperature/humidity averages, indoor vs. outdoor temperature/humidity gradients, dew point spread, and evening peak indicators.
- **Consensus Feature Selection:**
  - Majority-vote ranking combining Random Forest Gini importance, Recursive Feature Elimination (RFE) with Ridge, and target correlation.
- **Deep Recurrent Modeling (TensorFlow / Keras):**
  - 3D sliding temporal windows of shape `(n_samples, 36, n_features)` (6-hour historical lookback).
  - Target trained in variance-stabilised $\log(1 + y)$ space with exact inverse transformation back to Wh.
  - Stacked **LSTM**, **GRU**, and hybrid **CNN-LSTM** (1D causal convolution + pooling + LSTM) networks.
  - Early stopping with validation restoration and learning rate reduction on plateaus.
  - Random search hyperparameter tuning over units, dropout, learning rate, batch size, and optimizers.
- **Production Artifacts:**
  - Automated export of preprocessed datasets (`data/processed/`).
  - Serialized preprocessor pipeline and trained models (`models/`).
  - Standalone operational inference function for 10-minute next-step forecasting.

---

## Project Structure

```
energy_forecasting/
├── configs/
│   └── assessment.json               # Pipeline configuration
├── data/
│   ├── raw/
│   │   └── energy_data_set.csv       # Original UCI dataset (19,735 rows)
│   └── processed/                    # Exported preprocessed feature sets
│       ├── train_processed.csv
│       ├── val_processed.csv
│       └── test_processed.csv
├── models/                           # Persisted model and preprocessor artifacts
│   ├── best_model.keras              # Top-performing champion model
│   ├── lstm_model.keras              # Trained 2-layer LSTM
│   ├── gru_model.keras               # Trained 2-layer GRU
│   ├── cnn_lstm_model.keras          # Trained CNN-LSTM hybrid
│   ├── tuned_model.keras             # Tuned recurrent model
│   ├── ridge_model.joblib            # Ridge baseline
│   ├── random_forest_model.joblib    # Random Forest baseline
│   └── preprocessor.joblib           # Scalers, winsor fences & feature list
├── notebooks/
│   └── appliance_energy_prediction.ipynb # Complete 20-section master notebook
├── reports/                          # Metrics, predictions, and summaries
│   ├── final_model_comparison.csv
│   ├── test_predictions.csv
│   └── selected_features.json
├── src/                              # Modular Python package
│   ├── __init__.py
│   ├── data_preprocessing.py         # Loading, grid enforcement, winsorisation, scaling
│   ├── feature_engineering.py        # Causal lag/rolling features & consensus selection
│   ├── model.py                      # 3D sequences & Keras architectures (LSTM, GRU, CNN-LSTM)
│   ├── evaluate.py                   # Regression metrics (MAE, RMSE, MAPE, R2) & slice analysis
│   ├── train.py                      # End-to-end command-line training pipeline
│   └── predict.py                    # Standalone 10-minute next-step forecasting CLI
├── tests/
│   └── test_temporal_integrity.py    # Unit tests for causal integrity & alignment
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## Installation & Setup

Use Python 3.12:

```bash
git clone https://github.com/dilrukshax/energy_forecasting.git
cd energy_forecasting

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

### 1. Interactive Notebook
Open [`notebooks/appliance_energy_prediction.ipynb`](notebooks/appliance_energy_prediction.ipynb) in VS Code or JupyterLab.
Select the `.venv` kernel (`Python 3.12`) and run the cells sequentially to reproduce the full analysis, training, and evaluation.

### 2. Command-Line Training Pipeline
Run the complete automated pipeline (data loading, preprocessing, baseline fitting, deep learning, tuning, and artifact generation):

```bash
python -m src.train --data data/raw/energy_data_set.csv --output .
```

### 3. Production Inference / Forecasting
Forecast the next 10-minute energy consumption using the latest telemetry observations:

```bash
python -m src.predict --history data/raw/energy_data_set.csv
```

Example output:
```json
{
  "last_observed_timestamp": "2016-05-27 18:00:00",
  "forecast_timestamp": "2016-05-27 18:10:00",
  "forecast_horizon": "10 minutes",
  "predicted_energy_Wh": 64.21,
  "model_file": "best_model.keras",
  "features_used": 28
}
```

### 4. Running Unit Tests
Validate that there is no temporal target leakage and that sliding window alignments are exact:

```bash
python -m unittest discover -s tests -v
```

---

## Dataset Reference

Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). *Data driven prediction models of energy use of appliances in a low-energy house.* Energy and Buildings, 140, 81–97.  
Available on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

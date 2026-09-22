# Appliance energy forecasting

Predict the next 10-minute appliance energy reading in Wh using past energy,
temperature, humidity, lighting and calendar features. Compare LSTM and GRU
networks with persistence, daily seasonality, Ridge and Random Forest baselines.

The [notebook](notebooks/Assessment_Walkthrough.ipynb) contains the assessment
report: EDA, preprocessing, feature engineering, model design, optimization,
results and conclusions, with executed cells and embedded plots.

## Setup

Use Python 3.12. From the repository root:

```bash
git clone https://github.com/dilrukshax/energy_forecasting.git
cd energy_forecasting
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, create the environment with `py -3.12 -m venv .venv` and activate it
with `.venv\Scripts\Activate.ps1`. The pinned packages support the recorded CPU
experiment; a GPU is not required.

## Run

Open `notebooks/Assessment_Walkthrough.ipynb` in VS Code or Jupyter and select
the `.venv` Python kernel. Run all cells to reproduce preprocessing and inspect
the included results. Set `RUN_TRAINING = True` to train into `rerun/`.

The same training and inference steps are available from the command line:

```bash
python -m unittest discover -s tests -v
python -m src.predict --history data/raw/energy_data_set.csv
python -m src.train --output rerun
python -m src.predict --history data/raw/energy_data_set.csv --model-root rerun
```

To execute the report from the command line:

```bash
jupyter execute notebooks/Assessment_Walkthrough.ipynb --inplace --timeout=600
```

Choose a new output directory for each run. To deliberately replace the included
checkpoints, metrics and figures, use `python -m src.train --overwrite`, then
rerun the notebook with `RUN_TRAINING = False`.

## Method

- Chronological split: approximately 64% train, 16% validation, 20% test.
- Common 179-row warmup for all lag and sequence lengths.
- Features at time `t` use measurements through `t-1` and the known calendar at `t`.
- Median imputation, standardization, lag selection and feature ranking fit only
  training data. Real demand peaks are retained.
- Four recurrent configurations vary architecture, history length, units, layers,
  dropout, learning rate and batch size. Early stopping restores the best
  validation checkpoint; validation RMSE selects the final deep model.
- Test evaluation is rolling one-step forecasting. Earlier observed test readings
  may inform later predictions, but model weights remain fixed.

## Files

| Path | Contents |
| --- | --- |
| `data/raw/energy_data_set.csv` | Unchanged supplied data |
| `notebooks/Assessment_Walkthrough.ipynb` | Executed analysis and report |
| `src/` | Preprocessing, features, models, training, evaluation and inference |
| `configs/assessment.json` | Fixed experiment settings |
| `models/` | Saved models and preprocessing |
| `reports/` | Metrics, predictions, learning history and plots |
| `tests/` | Temporal leakage, missing-data and sequence-alignment checks |

Metrics are in [reports/test_metrics.csv](reports/test_metrics.csv); the runtime,
package versions and selected model are in
[reports/run_summary.json](reports/run_summary.json). Numerical results can vary
slightly across platforms. The notebook compares initial and tuned models without
assuming that a larger network improves performance.

## Dataset

Candanedo, L. (2017), [Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction),
UCI Machine Learning Repository. DOI: [10.24432/C5VC8G](https://doi.org/10.24432/C5VC8G).
Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The raw CSV is unchanged; engineered features are described in the notebook.

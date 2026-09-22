.PHONY: help install install-dev test test-fast lint format audit validate preprocess train train-fast predict runs docker clean

PYTHON ?= python

help:  ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package and its runtime dependencies
	$(PYTHON) -m pip install -e .

install-dev:  ## Install with the deep-learning and developer extras
	$(PYTHON) -m pip install -e ".[deep,dev]"

test:  ## Run the full test suite
	$(PYTHON) -m pytest

test-fast:  ## Run only the tests that do not touch the real dataset
	$(PYTHON) -m pytest -m "not slow"

lint:  ## Static checks
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m mypy src

format:  ## Auto-fix formatting and import order
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m ruff format src tests

validate:  ## Print the data quality report
	$(PYTHON) -m energy_forecast.cli validate

preprocess:  ## Materialise the processed dataset and design matrix into data/processed/
	$(PYTHON) scripts/preprocess.py

audit:  ## Print the serving-time feature availability audit
	$(PYTHON) -m energy_forecast.cli audit

train:  ## Full pipeline: baselines, deep models, tuning, figures, metrics
	$(PYTHON) -m energy_forecast.cli train

train-fast:  ## Baselines only; does not require TensorFlow
	$(PYTHON) -m energy_forecast.cli train --skip-deep

predict:  ## Score new data: make predict INPUT=data/raw/new.csv OUTPUT=reports/pred.csv
	$(PYTHON) scripts/predict.py --input $(INPUT) --output $(OUTPUT)

runs:  ## Compare every recorded run, best first
	$(PYTHON) -m energy_forecast.cli runs

docker:  ## Build the container image
	docker build -t energy-forecast .

clean:  ## Remove generated artefacts and caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf reports/figures/*.png reports/metrics.json reports/run.log
	rm -rf data/processed/*.parquet data/processed/*.csv
	rm -rf models/*.keras models/*.joblib
	find . -type d -name __pycache__ -exec rm -rf {} +

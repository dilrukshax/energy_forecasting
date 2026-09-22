"""Batch inference against saved artefacts.

Training that cannot be served is an experiment, not a system. This module closes the loop:
load the persisted model and the preprocessor that was fitted with it, rebuild features from
new raw data using the same code path as training, and emit predictions in Wh.

The critical property is that **inference reuses the training code**. It calls the same
:class:`~energy_forecast.features.FeatureBuilder` and the same fitted
:class:`~energy_forecast.preprocessing.Preprocessor`, so training/serving skew - the failure
where the two paths compute a feature slightly differently - is structurally impossible rather
than merely avoided by care.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from energy_forecast.config import Config, load_config
from energy_forecast.data import build_dataset
from energy_forecast.exceptions import ArtifactError, DataValidationError
from energy_forecast.features import FeatureBuilder
from energy_forecast.utils.logging import get_logger
from energy_forecast.features.preprocessing import Preprocessor
from energy_forecast.training.sequences import make_sequences

logger = get_logger(__name__)

MODEL_FILENAME = "best_model.keras"
PREPROCESSOR_FILENAME = "preprocessor.joblib"
FEATURES_FILENAME = "selected_features.json"


@dataclass
class ForecastService:
    """Loads the trained artefacts once and scores many batches.

    Args:
        config: Loaded configuration.
        model: The restored Keras model.
        preprocessor: The preprocessor fitted during training.
        selected_features: Feature names, in the order the model was trained on.
    """

    config: Config
    model: object
    preprocessor: Preprocessor
    selected_features: list

    @classmethod
    def load(cls, config: Optional[Config] = None,
             models_dir: Optional[Path] = None) -> "ForecastService":
        """Restore a service from the artefacts written by a training run.

        Raises:
            ArtifactError: if the model, the preprocessor or the feature list is missing.
        """
        import json

        config = config or load_config()
        directory = models_dir or config.path(config.outputs["models_dir"])

        model_path = directory / MODEL_FILENAME
        preprocessor_path = directory / PREPROCESSOR_FILENAME
        features_path = directory / FEATURES_FILENAME

        for path in (model_path, preprocessor_path, features_path):
            if not path.exists():
                raise ArtifactError(
                    f"missing artefact: {path}. Run `energy-forecast train` first."
                )

        from energy_forecast.models.architectures import load_model

        with open(features_path, encoding="utf-8") as handle:
            selected = json.load(handle)

        logger.info("loaded model and preprocessor from %s", directory)
        return cls(config=config, model=load_model(model_path),
                   preprocessor=Preprocessor.load(preprocessor_path),
                   selected_features=selected)

    def predict_frame(self, frame: pd.DataFrame) -> pd.Series:
        """Score a raw, time-indexed frame.

        The frame must contain the same sensor columns the model was trained on and cover
        enough history for the longest lag and the lookback window - otherwise the leading
        rows have no features and are dropped, which is reported rather than hidden.

        Args:
            frame: Raw observations, indexed by timestamp.

        Returns:
            Predicted consumption in Wh, indexed by the timestamps that could be scored.

        Raises:
            DataValidationError: if too little history is supplied to form a single window.
        """
        builder = FeatureBuilder(self.config)
        X, _ = builder.build(frame)

        missing = [name for name in self.selected_features if name not in X.columns]
        if missing:
            raise DataValidationError(
                f"input is missing {len(missing)} feature(s) the model needs, e.g. {missing[:5]}"
            )

        scaled = self.preprocessor.transform_features(X)
        positions = [self.preprocessor.feature_names.index(n) for n in self.selected_features]
        selected = scaled[:, positions]

        lookback = int(self.config.sequences["lookback"])
        if len(selected) <= lookback:
            raise DataValidationError(
                f"need more than {lookback} feature rows to form a window; "
                f"got {len(selected)}. Supply more history."
            )

        windows, _ = make_sequences(selected, np.zeros(len(selected)), lookback)
        predicted = self.preprocessor.target_transformer.inverse(
            self.model.predict(windows, verbose=0).ravel())

        index = X.index[lookback:]
        logger.info("scored %s timestamps (%s rows consumed as history)",
                    len(predicted), len(frame) - len(predicted))
        return pd.Series(predicted, index=index, name="predicted_wh")

    def predict_csv(self, csv_path: Path, output_path: Optional[Path] = None) -> pd.Series:
        """Score a raw CSV in the same format as the training data.

        Args:
            csv_path: Raw input, with the same schema as the training file.
            output_path: Optional destination for a two-column CSV of timestamp and prediction.

        Returns:
            Predicted consumption in Wh.
        """
        frame, _ = build_dataset(self.config, csv_path)
        predictions = self.predict_frame(frame)

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_frame().to_csv(output_path)
            logger.info("predictions written to %s", output_path)
        return predictions

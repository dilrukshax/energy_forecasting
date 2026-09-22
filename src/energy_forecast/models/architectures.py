"""Keras architectures and the training loop.

TensorFlow is imported lazily, inside the functions that need it. Importing it at module scope
would make the whole package - including the data, feature and split logic and their tests -
depend on a heavyweight optional dependency. ``pip install energy-forecast[deep]`` adds it.

Architecture choices and the reasoning behind them:

* **LSTM** - gated recurrence holds the daily level while still reacting to a sudden spike.
* **GRU** - one fewer gate, roughly 25% fewer parameters. With ~13k training windows the
  cheaper model may generalise better, which is worth measuring rather than assuming.
* **CNN-LSTM** - a causal 1-D convolution compresses local shape (the 10-30 minute ramp) before
  the LSTM models the longer dependency.

* ``tanh`` recurrent activation: the Keras default and the only setting that uses the fused
  cuDNN kernel; ReLU inside a recurrent loop tends to explode.
* ``relu`` dense head with a **linear output**: predictions live in standardised log space and
  must be free to go negative.
* **Adam**: adaptive per-parameter rates cope with mixed feature scales. RMSprop is in the
  search space rather than ruled out by assertion.
* **MSE loss**: the natural regression loss, and the one RMSE reports on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from energy_forecast.config import Config
from energy_forecast.utils.logging import get_logger

logger = get_logger(__name__)


def _keras():  # noqa: ANN202 - a thin lazy-import shim
    """Import Keras on demand, with an actionable error if it is absent."""
    try:
        from tensorflow import keras
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "TensorFlow is required for the deep models. "
            "Install it with: pip install -e '.[deep]'"
        ) from exc
    return keras


def set_seeds(config: Config) -> None:
    """Seed NumPy, Python and TensorFlow so a run is reproducible."""
    import random

    seed = config.random_state
    random.seed(seed)
    np.random.seed(seed)
    try:
        keras = _keras()
        keras.utils.set_random_seed(seed)
    except ImportError:  # pragma: no cover
        logger.warning("TensorFlow unavailable; only NumPy and Python seeded")


@dataclass(frozen=True)
class Hyperparameters:
    """One point in the search space."""

    units: int = 64
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 64
    optimizer: str = "adam"

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {"units": self.units, "dropout": self.dropout,
                "learning_rate": self.learning_rate, "batch_size": self.batch_size,
                "optimizer": self.optimizer}


def _optimizer(name: str, learning_rate: float):  # noqa: ANN202
    """Build an optimiser by name."""
    keras = _keras()
    if name == "adam":
        return keras.optimizers.Adam(learning_rate)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate)
    raise ValueError(f"unsupported optimizer: {name!r}")


def build_lstm(input_shape: Tuple[int, int], params: Hyperparameters):  # noqa: ANN201
    """Two stacked LSTM layers with dropout and a small dense head."""
    keras = _keras()
    layers = keras.layers
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(params.units, return_sequences=True, dropout=params.dropout),
        layers.LSTM(max(params.units // 2, 8), dropout=params.dropout),
        layers.Dense(32, activation="relu"),
        layers.Dropout(params.dropout),
        layers.Dense(1),
    ], name="lstm")
    model.compile(optimizer=_optimizer(params.optimizer, params.learning_rate),
                  loss="mse", metrics=["mae"])
    return model


def build_gru(input_shape: Tuple[int, int], params: Hyperparameters):  # noqa: ANN201
    """GRU counterpart of the LSTM, same depth and head."""
    keras = _keras()
    layers = keras.layers
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(params.units, return_sequences=True, dropout=params.dropout),
        layers.GRU(max(params.units // 2, 8), dropout=params.dropout),
        layers.Dense(32, activation="relu"),
        layers.Dropout(params.dropout),
        layers.Dense(1),
    ], name="gru")
    model.compile(optimizer=_optimizer(params.optimizer, params.learning_rate),
                  loss="mse", metrics=["mae"])
    return model


def build_cnn_lstm(input_shape: Tuple[int, int], params: Hyperparameters):  # noqa: ANN201
    """Causal convolution over local shape, then an LSTM over the longer dependency."""
    keras = _keras()
    layers = keras.layers
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(params.units, kernel_size=3, padding="causal", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.LSTM(params.units, dropout=params.dropout),
        layers.Dense(32, activation="relu"),
        layers.Dropout(params.dropout),
        layers.Dense(1),
    ], name="cnn_lstm")
    model.compile(optimizer=_optimizer(params.optimizer, params.learning_rate),
                  loss="mse", metrics=["mae"])
    return model


ARCHITECTURES: Dict[str, Callable[[Tuple[int, int], Hyperparameters], Any]] = {
    "lstm": build_lstm,
    "gru": build_gru,
    "cnn_lstm": build_cnn_lstm,
}


def build_model(architecture: str, input_shape: Tuple[int, int], params: Hyperparameters):  # noqa: ANN201
    """Dispatch to an architecture by name.

    Raises:
        KeyError: if the architecture is not registered.
    """
    if architecture not in ARCHITECTURES:
        raise KeyError(f"unknown architecture {architecture!r}; "
                       f"available: {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[architecture](input_shape, params)


def train(model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
          y_val: np.ndarray, params: Hyperparameters, epochs: int,
          early_stopping_patience: int, reduce_lr_patience: int, verbose: int = 0):  # noqa: ANN201
    """Fit with early stopping on validation loss.

    ``restore_best_weights=True`` means the returned model carries the weights from the best
    validation epoch, not the last one - without it, early stopping merely stops late.

    ``shuffle=False`` keeps mini-batches chronologically coherent. With pre-windowed data it is
    not required for correctness, but it keeps runs reproducible.

    Returns:
        The Keras ``History``.
    """
    keras = _keras()
    callbacks: List[Any] = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=early_stopping_patience,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=reduce_lr_patience, min_lr=1e-5),
    ]
    return model.fit(X_train, y_train, validation_data=(X_val, y_val),
                     epochs=epochs, batch_size=params.batch_size,
                     callbacks=callbacks, shuffle=False, verbose=verbose)


def save_model(model, path: Path) -> Path:
    """Persist a trained model in the native Keras format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    logger.info("model saved to %s", path)
    return path


def load_model(path: Path):  # noqa: ANN201
    """Load a persisted Keras model."""
    return _keras().models.load_model(path)

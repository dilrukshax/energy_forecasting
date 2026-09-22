"""TensorFlow / Keras deep learning architectures and sequence generation for energy forecasting."""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DEFAULT_LOOKBACK = 36  # 6 hours at 10-minute intervals


def make_sequences(features, targets, lookback=DEFAULT_LOOKBACK):
    """Transform a 2-D feature matrix and target series into 3-D sliding windows.

    Returns X of shape (n_windows, lookback, n_features) and y of shape (n_windows,), where
    X[i] spans rows i .. i+lookback-1 and y[i] is the target at row i+lookback - so each window
    ends strictly before the value it predicts.
    """
    windows = np.lib.stride_tricks.sliding_window_view(
        features, (lookback, features.shape[1])
    ).squeeze(1)[:-1]
    return np.ascontiguousarray(windows), np.asarray(targets)[lookback:]


def build_lstm(input_shape, units=64, dropout=0.2, lr=1e-3):
    """Two stacked LSTM layers with dropout and a small dense head."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(units, return_sequences=True, dropout=dropout),
        layers.LSTM(max(units // 2, 8), dropout=dropout),
        layers.Dense(32, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(1),
    ], name="LSTM")
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return model


def build_gru(input_shape, units=64, dropout=0.2, lr=1e-3):
    """GRU counterpart of the LSTM with identical depth and head."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(units, return_sequences=True, dropout=dropout),
        layers.GRU(max(units // 2, 8), dropout=dropout),
        layers.Dense(32, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(1),
    ], name="GRU")
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return model


def build_cnn_lstm(input_shape, units=64, dropout=0.2, lr=1e-3, filters=None):
    """1-D causal convolutional front end (local temporal patterns) followed by LSTM."""
    filters = filters or units
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(filters, kernel_size=3, padding="causal", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.LSTM(units, dropout=dropout),
        layers.Dense(32, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(1),
    ], name="CNN_LSTM")
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return model


BUILDERS = {
    "LSTM": build_lstm,
    "GRU": build_gru,
    "CNN-LSTM": build_cnn_lstm
}


def build_tuned(arch, input_shape, units, dropout, lr, optimizer="adam"):
    """Rebuild the chosen architecture with custom hyper-parameter settings."""
    builder = BUILDERS[arch]
    model = builder(input_shape=input_shape, units=units, dropout=dropout, lr=lr)
    opt = keras.optimizers.Adam(lr) if optimizer == "adam" else keras.optimizers.RMSprop(lr)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=60, batch_size=64, patience=8, verbose=0):
    """Fit on the training windows, early-stop on validation loss, and restore the best weights."""
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
        ),
    ]
    return model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, shuffle=False, verbose=verbose
    )

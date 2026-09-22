"""Unit tests for temporal integrity, causal feature engineering, and sequence generation."""
import unittest
import numpy as np
import pandas as pd

from src.data_preprocessing import (
    load_data, calculate_winsor_bounds, winsorise, TargetScaler
)
from src.feature_engineering import build_features
from src.model import make_sequences


class TemporalIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_data("data/raw/energy_data_set.csv")

    def test_future_measurements_do_not_leak_into_past_features(self):
        """Verifies that altering data at and after time t does not change features <= t."""
        # Use first 2500 rows (sufficient for 1-week warm-up of 1008 steps)
        sub = self.frame.iloc[:2500].copy()
        X_orig, _ = build_features(sub)

        # Alter all measurements at and after the midpoint of the usable range
        mid_idx = len(X_orig) // 2
        cutoff_date = X_orig.index[mid_idx]

        sub_altered = sub.copy()
        sub_altered.loc[cutoff_date:, :] = 99999.0

        X_altered, _ = build_features(sub_altered)

        # Features strictly before the alteration must be exactly identical
        pd.testing.assert_frame_equal(
            X_orig.loc[:cutoff_date],
            X_altered.loc[:cutoff_date]
        )

    def test_lag_alignment(self):
        """Verifies that app_lag1 at row t equals target at row t-1."""
        X, y = build_features(self.frame.iloc[:2000])
        idx_t = X.index[200]
        prev_idx = self.frame.index[self.frame.index.get_loc(idx_t) - 1]

        self.assertEqual(X.loc[idx_t, "app_lag1"], self.frame.loc[prev_idx, "Appliances"])

    def test_winsor_bounds_use_only_training_data(self):
        """Verifies that winsorisation fences computed on train are independent of test data."""
        X, _ = build_features(self.frame.iloc[:2000])
        n_train = len(X) // 2
        X_train = X.iloc[:n_train]

        bounds_original = calculate_winsor_bounds(X_train, k=3.0)

        # Create altered dataset where holdout is corrupted
        X_corrupted = X.copy()
        X_corrupted.iloc[n_train:] = 1e8
        bounds_from_corrupted = calculate_winsor_bounds(X_corrupted.iloc[:n_train], k=3.0)

        for col in bounds_original:
            self.assertEqual(bounds_original[col], bounds_from_corrupted[col])

    def test_sequence_windows_do_not_include_target(self):
        """Verifies that window i ends strictly at row i+lookback-1 and target is at i+lookback."""
        features = np.arange(100, dtype=np.float32).reshape(50, 2)
        targets = np.arange(50, dtype=np.float32) * 10
        lookback = 6

        X_seq, y_seq = make_sequences(features, targets, lookback=lookback)

        self.assertEqual(X_seq.shape, (50 - lookback, lookback, 2))
        self.assertEqual(y_seq.shape, (50 - lookback,))

        # Window 0 spans rows 0..5, target is at row 6
        np.testing.assert_array_equal(X_seq[0], features[0:6])
        self.assertEqual(y_seq[0], targets[6])

    def test_target_scaler_roundtrip(self):
        """Verifies that TargetScaler accurately roundtrips positive values."""
        y = np.array([10.0, 50.0, 100.0, 650.0, 1080.0])
        scaler = TargetScaler().fit(y)
        y_scaled = scaler.transform(y)
        y_recovered = scaler.inverse_transform(y_scaled)

        np.testing.assert_allclose(y, y_recovered, atol=1e-5)


if __name__ == "__main__":
    unittest.main()

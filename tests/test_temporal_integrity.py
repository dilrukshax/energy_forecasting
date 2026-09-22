"""Tests for the material risks: target leakage, fitting on holdout, and alignment."""
import unittest
import numpy as np
import pandas as pd
from src.data_preprocessing import load_data, split_boundaries, fit_feature_preprocessing
from src.feature_engineering import make_features, BASE_LAGS, WARMUP, MAX_LAG
from src.model import SequenceDataset


class TemporalIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_data("data/raw/energy_data_set.csv")

    def test_current_and_future_measurements_cannot_change_current_features(self):
        t = 400
        before = make_features(self.frame.iloc[:700], BASE_LAGS)
        changed = self.frame.iloc[:700].copy()
        changed.iloc[t:, :] = 99999.0
        after = make_features(changed, BASE_LAGS)
        pd.testing.assert_frame_equal(before.iloc[:t+1], after.iloc[:t+1])

    def test_energy_window_has_exact_past_values(self):
        f = make_features(self.frame, BASE_LAGS)
        t = 400
        self.assertEqual(f.energy_lag_1.iloc[t], self.frame.Appliances.iloc[t-1])
        self.assertEqual(f.energy_lag_3.iloc[t], self.frame.Appliances.iloc[t-3])
        self.assertAlmostEqual(f.energy_mean_6.iloc[t], self.frame.Appliances.iloc[t-6:t].mean())
        self.assertNotIn("Appliances", f.columns)
        self.assertNotIn("rv1", f.columns)
        self.assertNotIn("T_out_lag1", f.columns)

    def test_training_transform_statistics_do_not_depend_on_holdout(self):
        end, _ = split_boundaries(len(self.frame))
        f = make_features(self.frame, BASE_LAGS)
        altered = f.copy()
        altered.iloc[end:] = 1e9
        fitted = []
        for candidate in [f, altered]:
            _, imputer, scaler = fit_feature_preprocessing(candidate, end, MAX_LAG)
            fitted.append((imputer.statistics_, scaler.mean_, scaler.scale_))
        for original, changed in zip(*fitted):
            np.testing.assert_array_equal(original, changed)

    def test_missing_sensor_values_use_only_past_readings(self):
        frame = self.frame.iloc[:700].copy()
        frame.iloc[400:410, frame.columns.get_loc("T1")] = np.nan
        features = make_features(frame, BASE_LAGS)
        self.assertEqual(features.T1_lag1.iloc[406], frame.T1.iloc[399])
        self.assertTrue(np.isnan(features.T1_lag1.iloc[407]))
        altered = frame.copy()
        altered.iloc[410:, altered.columns.get_loc("T1")] = 99999.0
        pd.testing.assert_frame_equal(features.iloc[:411], make_features(altered, BASE_LAGS).iloc[:411])
        scaled, imputer, _ = fit_feature_preprocessing(features, 500, MAX_LAG)
        self.assertTrue(np.isfinite(scaled).all())
        position = features.columns.get_loc("T1_lag1")
        self.assertEqual(imputer.statistics_[position], features.T1_lag1.iloc[MAX_LAG:500].median())

    def test_all_missing_training_feature_is_rejected(self):
        features = make_features(self.frame.iloc[:700], BASE_LAGS)
        features.loc[:, "T1_lag1"] = np.nan
        with self.assertRaisesRegex(ValueError, "observed training values"):
            fit_feature_preprocessing(features, 500, MAX_LAG)

    def test_sequence_target_and_split_boundaries(self):
        train_end, test_start = split_boundaries(len(self.frame))
        self.assertLess(WARMUP, train_end)
        self.assertLess(train_end, test_start)
        x = np.arange(1000, dtype=np.float32).reshape(500, 2)
        y = np.arange(500, dtype=np.float32)
        dataset = SequenceDataset(x, y, [200], lookback=18)
        sequence, target = dataset[0]
        np.testing.assert_array_equal(sequence.numpy(), x[183:201])
        self.assertEqual(target.item(), y[200])


if __name__ == "__main__":
    unittest.main()

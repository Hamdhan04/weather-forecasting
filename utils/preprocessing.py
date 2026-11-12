"""
Data Preprocessing and Pipeline Utilities for Time-Series LSTM Modeling.
Features:
- MinMaxScaler normalization (preventing data leakage by fitting on train split only)
- Chronological Train / Validation / Test dataset partitioning
- Multi-feature Sliding Window generator for Sequence-to-Sequence / Sequence-to-Horizon forecasting
"""

import os
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib


FEATURE_COLUMNS = ["temperature", "humidity", "pressure", "wind_speed"]
TARGET_COLUMN = "temperature"


class WeatherDataPipeline:
    """
    Standardizes weather data preparation, feature scaling, and lookback window creation.
    """

    def __init__(
        self,
        lookback_window: int = 48,
        horizon: int = 24,
        features: Optional[List[str]] = None,
        target: str = TARGET_COLUMN,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15
    ):
        self.lookback_window = lookback_window
        self.horizon = horizon
        self.features = features or FEATURE_COLUMNS
        self.target = target
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_idx = self.features.index(self.target)

    def fit_transform_splits(
        self,
        df: pd.DataFrame
    ) -> Tuple[
        np.ndarray, np.ndarray,
        np.ndarray, np.ndarray,
        np.ndarray, np.ndarray,
        pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex
    ]:
        """
        Partition dataframe chronologically into Train, Val, Test, fit scalers,
        and construct sliding window tensors (X, y).
        """
        # Ensure timestamp sorting
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)
            timestamps = pd.to_datetime(df["timestamp"])
        else:
            timestamps = pd.date_range("2022-01-01", periods=len(df), freq="h")

        raw_features = df[self.features].values
        target_series = df[[self.target]].values

        n_samples = len(df)
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.val_ratio))

        # Train partition
        train_feats = raw_features[:train_end]
        train_target = target_series[:train_end]

        # Fit scalers strictly on training data
        self.feature_scaler.fit(train_feats)
        self.target_scaler.fit(train_target)

        # Scale all sets
        scaled_features = self.feature_scaler.transform(raw_features)
        scaled_target = self.target_scaler.transform(target_series).ravel()

        # Sliding window construction
        X, y, window_times = self._create_windows(scaled_features, scaled_target, timestamps)

        # Splitting windowed data according to chronological indices
        n_windows = len(X)
        w_train_end = int(n_windows * self.train_ratio)
        w_val_end = int(n_windows * (self.train_ratio + self.val_ratio))

        X_train, y_train = X[:w_train_end], y[:w_train_end]
        X_val, y_val = X[w_train_end:w_val_end], y[w_train_end:w_val_end]
        X_test, y_test = X[w_val_end:], y[w_val_end:]

        time_train = window_times[:w_train_end]
        time_val = window_times[w_train_end:w_val_end]
        time_test = window_times[w_val_end:]

        return (
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            time_train, time_val, time_test
        )

    def _create_windows(
        self,
        features: np.ndarray,
        target: np.ndarray,
        timestamps: pd.DatetimeIndex
    ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Creates (N, lookback, num_features) and (N, horizon) sliding sequences.
        """
        X, y = [], []
        target_times = []
        total_len = len(features)

        for i in range(total_len - self.lookback_window - self.horizon + 1):
            x_seq = features[i : i + self.lookback_window]
            y_seq = target[i + self.lookback_window : i + self.lookback_window + self.horizon]
            X.append(x_seq)
            y.append(y_seq)
            # Timestamp corresponds to the start of the forecasted horizon
            target_times.append(timestamps[i + self.lookback_window])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), pd.DatetimeIndex(target_times)

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Convert normalized target predictions back to original physical units (°C).
        """
        shape = y_scaled.shape
        flat = y_scaled.reshape(-1, 1)
        inv = self.target_scaler.inverse_transform(flat)
        return inv.reshape(shape)

    def transform_single_window(self, input_df: pd.DataFrame) -> np.ndarray:
        """
        Transforms a dataframe of shape (lookback_window, n_features) for real-time inference.
        """
        feats = input_df[self.features].values
        scaled = self.feature_scaler.transform(feats)
        return np.expand_dims(scaled, axis=0).astype(np.float32)

    def save_scalers(self, directory: str = "model"):
        """Save fitted scalers for dashboard reuse."""
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.feature_scaler, os.path.join(directory, "feature_scaler.pkl"))
        joblib.dump(self.target_scaler, os.path.join(directory, "target_scaler.pkl"))

    def load_scalers(self, directory: str = "model"):
        """Load pre-fitted scalers."""
        feat_path = os.path.join(directory, "feature_scaler.pkl")
        target_path = os.path.join(directory, "target_scaler.pkl")
        if os.path.exists(feat_path) and os.path.exists(target_path):
            self.feature_scaler = joblib.load(feat_path)
            self.target_scaler = joblib.load(target_path)
            return True
        return False

"""
Stacked LSTM Neural Network Architecture for Multi-Variate Weather Time-Series Forecasting.
Specification:
- Input dimension: (batch_size, lookback_window, n_features)
- LSTM Layer 1: 64 hidden units (return_sequences=True) + Dropout(0.2)
- LSTM Layer 2: 32 hidden units (return_sequences=False) + Dropout(0.2)
- Dense Layer: Linear projection to target horizon predictions
"""

import os
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np


class WeatherStackedLSTM(nn.Module):
    """
    Two-tier Stacked LSTM network with recurrent dropout and dense linear readout.
    """
    def __init__(
        self,
        input_size: int = 4,      # Temperature, Humidity, Pressure, Wind Speed
        hidden_dim1: int = 64,    # LSTM Layer 1 units
        hidden_dim2: int = 32,    # LSTM Layer 2 units
        horizon: int = 24,        # Forecast horizon length
        dropout: float = 0.2
    ):
        super(WeatherStackedLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.horizon = horizon

        # LSTM Layer 1: processes input sequence and outputs sequences
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim1,
            num_layers=1,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(p=dropout)

        # LSTM Layer 2: processes intermediate sequences and outputs final state
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim1,
            hidden_size=hidden_dim2,
            num_layers=1,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(p=dropout)

        # Fully connected readout head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim2, 16),
            nn.ReLU(),
            nn.Linear(16, horizon)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through stacked LSTM layers.
        x: Tensor of shape (batch_size, seq_len, input_size)
        returns: Tensor of shape (batch_size, horizon)
        """
        # LSTM 1 (return_sequences=True)
        out1, _ = self.lstm1(x)
        out1 = self.dropout1(out1)

        # LSTM 2 (return_sequences=False -> select final timestep output)
        out2, _ = self.lstm2(out1)
        last_step = out2[:, -1, :]  # Shape: (batch_size, hidden_dim2)
        last_step = self.dropout2(last_step)

        # Readout head
        predictions = self.fc(last_step)
        return predictions

    def predict_numpy(self, x_np: np.ndarray, device: str = "cpu") -> np.ndarray:
        """
        Run inference on a numpy array of shape (N, lookback, features).
        """
        self.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(x_np).float().to(device)
            out = self.forward(tensor_x)
            return out.cpu().numpy()

    def save(self, filepath: str, extra_meta: Optional[Dict[str, Any]] = None):
        """Save model weights and architecture hyperparameters."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "config": {
                "input_size": self.input_size,
                "hidden_dim1": self.hidden_dim1,
                "hidden_dim2": self.hidden_dim2,
                "horizon": self.horizon
            },
            "meta": extra_meta or {}
        }
        torch.save(payload, filepath)

    @classmethod
    def load(cls, filepath: str, device: str = "cpu"):
        """Load trained model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=device)
        config = checkpoint.get("config", {"input_size": 4, "hidden_dim1": 64, "hidden_dim2": 32, "horizon": 24})
        model = cls(
            input_size=config["input_size"],
            hidden_dim1=config["hidden_dim1"],
            hidden_dim2=config["hidden_dim2"],
            horizon=config["horizon"]
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model, checkpoint.get("meta", {})

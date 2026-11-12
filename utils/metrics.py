"""
Evaluation Metrics for Time-Series Regression & Forecasting.
Calculates RMSE, MAE, R-Squared Score, and Mean Absolute Percentage Error (MAPE).
"""

from typing import Dict
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute key academic evaluation metrics for regression models.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth meteorological targets.
    y_pred : np.ndarray
        Model forecasted values.
        
    Returns:
    --------
    Dict[str, float]:
        Calculated metrics (RMSE, MAE, R2, MAPE, Max Error).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Avoid zero division in MAPE
    mask = np.abs(y_true) > 1e-3
    if np.any(mask):
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    else:
        mape = 0.0

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    max_err = float(np.max(np.abs(y_true - y_pred)))

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "mape": round(mape, 2),
        "max_error": round(max_err, 4)
    }

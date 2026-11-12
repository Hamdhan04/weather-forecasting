"""
Training Pipeline for Weather Forecasting Stacked LSTM.
Handles data loading, sequence preparation, model training, validation tracking,
metric evaluation, and model artifact serialization.
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.weather_generator import generate_synthetic_weather_data
from utils.preprocessing import WeatherDataPipeline, FEATURE_COLUMNS, TARGET_COLUMN
from utils.metrics import calculate_metrics
from model.lstm_model import WeatherStackedLSTM


def train_model(
    data_path: str = "data/weather_data.csv",
    lookback_window: int = 48,
    horizon: int = 48,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 0.001,
    seed: int = 42,
    save_dir: str = "model"
):
    """
    Executes end-to-end model training, validation, and evaluation.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    # 1. Load or generate dataset
    if not os.path.exists(data_path):
        print(f"[Notice] Dataset '{data_path}' not found. Generating realistic synthetic dataset...")
        df = generate_synthetic_weather_data(output_path=data_path, n_days=730, seed=seed)
    else:
        print(f"[Data] Loading dataset from: {data_path}")
        df = pd.read_csv(data_path)

    # 2. Preprocessing & Partitioning
    print("[Pipeline] Transforming features and creating sliding window tensors...")
    pipeline = WeatherDataPipeline(
        lookback_window=lookback_window,
        horizon=horizon,
        features=FEATURE_COLUMNS,
        target=TARGET_COLUMN
    )

    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        time_train, time_val, time_test
    ) = pipeline.fit_transform_splits(df)

    # Save fitted scalers
    pipeline.save_scalers(save_dir)
    print(f"[Pipeline] Windowed Samples -> Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

    # PyTorch DataLoaders
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Model Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Model] Initializing Stacked LSTM (Input: {len(FEATURE_COLUMNS)}, LSTM-1: 64, LSTM-2: 32, Horizon: {horizon}) on {device}")

    model = WeatherStackedLSTM(
        input_size=len(FEATURE_COLUMNS),
        hidden_dim1=64,
        hidden_dim2=32,
        horizon=horizon,
        dropout=0.2
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # 4. Training Loop
    history = {"train_loss": [], "val_loss": [], "epochs": []}
    best_val_loss = float("inf")
    best_weights_path = os.path.join(save_dir, "saved_model.pt")

    start_time = time.time()
    print("\n" + "=" * 50)
    print("Beginning Model Training")
    print("=" * 50)

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item() * len(batch_x)

        epoch_train_loss = running_train_loss / len(train_dataset)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                val_preds = model(batch_x)
                v_loss = criterion(val_preds, batch_y)
                running_val_loss += v_loss.item() * len(batch_x)

        epoch_val_loss = running_val_loss / len(val_dataset)
        scheduler.step(epoch_val_loss)

        history["train_loss"].append(round(epoch_train_loss, 6))
        history["val_loss"].append(round(epoch_val_loss, 6))
        history["epochs"].append(epoch)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss (MSE): {epoch_train_loss:.6f} | Val Loss (MSE): {epoch_val_loss:.6f}")

        # Checkpoint best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            model.save(best_weights_path)

    training_duration = round(time.time() - start_time, 2)
    print(f"\n[Training Complete] Elapsed time: {training_duration}s | Best Val Loss: {best_val_loss:.6f}")

    # 5. Final Evaluation on Test Partition
    print("[Evaluation] Computing test partition predictions and unscaling...")
    best_model, _ = WeatherStackedLSTM.load(best_weights_path, device=device)
    test_preds_scaled = best_model.predict_numpy(X_test, device=str(device))

    # Inverse transform to degrees Celsius (°C)
    y_test_unscaled = pipeline.inverse_transform_target(y_test)
    test_preds_unscaled = pipeline.inverse_transform_target(test_preds_scaled)

    # 1-step ahead metrics
    metrics_1step = calculate_metrics(y_test_unscaled[:, 0], test_preds_unscaled[:, 0])
    # 24-step ahead metrics
    metrics_24step = calculate_metrics(y_test_unscaled[:, min(23, horizon-1)], test_preds_unscaled[:, min(23, horizon-1)])
    # Overall multi-horizon metrics
    metrics_overall = calculate_metrics(y_test_unscaled, test_preds_unscaled)

    print("\n" + "=" * 50)
    print("Test Set Performance Summary")
    print("=" * 50)
    print(f"  RMSE (Root Mean Sq Error): {metrics_overall['rmse']} °C")
    print(f"  MAE  (Mean Absolute Error): {metrics_overall['mae']} °C")
    print(f"  R²   (Determination Score): {metrics_overall['r2']}")
    print(f"  MAPE (Mean Abs % Error)  : {metrics_overall['mape']}%")
    print("=" * 50)

    # 6. Save comprehensive evaluation metadata for Dashboard consumption
    meta_path = os.path.join(save_dir, "model_metadata.json")
    metadata = {
        "architecture": {
            "type": "Stacked LSTM",
            "layers": [
                {"name": "LSTM Layer 1", "units": 64, "return_sequences": True, "dropout": 0.2},
                {"name": "LSTM Layer 2", "units": 32, "return_sequences": False, "dropout": 0.2},
                {"name": "Dense Linear Head", "projection": horizon}
            ],
            "lookback_window": lookback_window,
            "horizon": horizon,
            "features": FEATURE_COLUMNS,
            "target": TARGET_COLUMN
        },
        "training": {
            "epochs_run": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "duration_seconds": training_duration,
            "history": history
        },
        "metrics": {
            "overall": metrics_overall,
            "step_1": metrics_1step,
            "step_24": metrics_24step
        },
        "sample_test_timestamps": [t.isoformat() for t in time_test[-200:]],
        "sample_test_actual": y_test_unscaled[-200:, 0].tolist(),
        "sample_test_predicted": test_preds_unscaled[-200:, 0].tolist()
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Artifacts] Model saved to '{best_weights_path}'")
    print(f"[Artifacts] Metadata saved to '{meta_path}'")
    return model, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Weather Stacked LSTM Model")
    parser.add_argument("--data", type=str, default="data/weather_data.csv", help="Path to weather dataset CSV")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lookback", type=int, default=48, help="Lookback window size in hours")
    parser.add_argument("--horizon", type=int, default=48, help="Prediction horizon in hours")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    train_model(
        data_path=args.data,
        lookback_window=args.lookback,
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

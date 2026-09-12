"""
Model 1: Rainfall Prediction & Downscaling Neural Network
Predicts gridded daily precipitation across India from 5 ERA5 atmospheric
variables (tp, t2m, d2m, u10, v10).

Includes:
- SpatiotemporalRainfallNet (2D Fully Convolutional Residual Regressor)
- MaskedHuberLoss / MaskedMSELoss (evaluates exclusively over Indian land territory)
- Tabular Gradient Boosted Trees baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SpatiotemporalRainfallNet(nn.Module):
    """
    Fully Convolutional Residual Network for meteorological grid precipitation estimation.
    Preserves spatial coordinate dimensions (117 x 117).
    """
    def __init__(self, in_channels: int = 5, base_channels: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        
        # Dilated convolution layer to expand receptive field across monsoon fronts
        self.dilated = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.dec2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels)
        
        self.out_head = nn.Sequential(
            nn.Conv2d(base_channels, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.ReLU() # Rainfall (log1p) is strictly non-negative
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        d = self.dilated(e3)
        
        cat2 = torch.cat([d, e2], dim=1)
        u2 = self.dec2(cat2)
        
        cat1 = torch.cat([u2, e1], dim=1)
        u1 = self.dec1(cat1)
        
        out = self.out_head(u1)
        return out

class MaskedLoss(nn.Module):
    """
    Computes Loss strictly on valid Indian land cells, ignoring ocean/foreign grid cells.
    """
    def __init__(self, loss_type="huber", delta=1.0):
        super().__init__()
        self.loss_type = loss_type
        self.delta = delta

    def forward(self, pred, target, land_mask):
        # pred: [B, 1, H, W], target: [B, 1, H, W], land_mask: [H, W]
        mask = land_mask.unsqueeze(0).unsqueeze(0).expand_as(pred)
        pred_land = pred[mask]
        target_land = target[mask]
        
        if self.loss_type == "huber":
            return F.huber_loss(pred_land, target_land, delta=self.delta)
        else:
            return F.mse_loss(pred_land, target_land)

def train_tabular_rainfall_model(csv_path: str, max_samples: int = 100000):
    """
    Trains a high-performance HistGradientBoostingRegressor on tabular meteorological features.
    """
    print(f"Loading tabular rainfall dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    features = [
        "lat", "lon", "doy_sin", "doy_cos",
        "tp_era5", "t2m", "d2m", "u10", "v10",
        "wind_speed", "dew_depression"
    ]
    target = "rainfall_imd"

    # Split chronologically
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    split_idx = int(len(df) * 0.8)

    X_train, y_train = df[features].iloc[:split_idx], df[target].iloc[:split_idx]
    X_test, y_test = df[features].iloc[split_idx:], df[target].iloc[split_idx:]

    print(f"Training HistGradientBoostingRegressor on {len(X_train)} samples...")
    model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    preds = np.maximum(preds, 0.0)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Tabular Rainfall Model Test Metrics:")
    print(f"  RMSE: {rmse:.3f} mm")
    print(f"  MAE:  {mae:.3f} mm")
    print(f"  R^2:  {r2:.3f}")

    return model, {"rmse": rmse, "mae": mae, "r2": r2}

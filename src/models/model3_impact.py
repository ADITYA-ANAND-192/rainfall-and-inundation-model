"""
Model 3: Infrastructure Impact & Disaster Risk Assessment Model
Takes inputs from:
- Model 1 (Rainfall volume / intensity)
- Model 2 (Flood inundation extent / % flooded area)
- District exposure indicators (population, permanent water, duration)

Predicts:
1. Human Fatalities (continuous)
2. Human Injured (continuous)
3. Infrastructure Risk Score (0 - 100)
4. Disaster Severity Level (Low, Moderate, High, Critical)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ImpactMultiTaskNet(nn.Module):
    """
    Multi-Task Neural Network for compound disaster impact prediction.
    Outputs: [B, 3] -> (log1p_fatalities, log1p_injured, infrastructure_risk_score)
    """
    def __init__(self, in_features: int = 8, hidden_dim: int = 64):
        super().__init__()
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Head 1: Fatalities Head
        self.fatality_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU() # Non-negative
        )

        # Head 2: Injured Head
        self.injured_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU() # Non-negative
        )

        # Head 3: Infrastructure Risk Score Head (0 - 100)
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Will scale by 100.0
        )

    def forward(self, x):
        feat = self.shared(x)
        fatality = self.fatality_head(feat)
        injured = self.injured_head(feat)
        risk = self.risk_head(feat) * 100.0
        return torch.cat([fatality, injured, risk], dim=-1)

class MultiTaskImpactLoss(nn.Module):
    """
    Weighted multi-task loss combining MSE for casualties and Huber loss for risk score.
    """
    def __init__(self, w_fatality=1.0, w_injured=0.8, w_risk=0.01):
        super().__init__()
        self.w_fatality = w_fatality
        self.w_injured = w_injured
        self.w_risk = w_risk

    def forward(self, preds, targets):
        # preds: [B, 3], targets: [B, 3]
        loss_fat = F.huber_loss(preds[:, 0], targets[:, 0])
        loss_inj = F.huber_loss(preds[:, 1], targets[:, 1])
        loss_rsk = F.mse_loss(preds[:, 2], targets[:, 2])

        total_loss = self.w_fatality * loss_fat + self.w_injured * loss_inj + self.w_risk * loss_rsk
        return total_loss, {"loss_fatality": loss_fat.item(), "loss_injured": loss_inj.item(), "loss_risk": loss_rsk.item()}

def train_tabular_impact_baseline(
    train_csv: str = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\processed_data\district_features\train_impact.csv",
    test_csv: str = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\processed_data\district_features\test_impact.csv"
):
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    feature_cols = [
        "Percent_Flooded_Area", "Parmanent_Water", "Corrected_Percent_Flooded_Area",
        "Population", "Mean_Flood_Duration", "Flooded_Population_Proxy",
        "Hazard_Exposure_Product", "Duration_Weighted_Inundation"
    ]

    target_cols = ["Human_fatality", "Human_injured", "Infrastructure_Risk_Score"]

    X_train = df_train[feature_cols].copy()
    X_test = df_test[feature_cols].copy()
    X_train["Population"] = np.log1p(X_train["Population"])
    X_test["Population"] = np.log1p(X_test["Population"])

    models = {}
    metrics = {}

    for tgt in target_cols:
        y_train = df_train[tgt]
        y_test = df_test[tgt]

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        preds = rf.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        models[tgt] = rf
        metrics[tgt] = {"rmse": rmse, "r2": r2}
        print(f"Target '{tgt}': Test RMSE = {rmse:.2f}, R^2 = {r2:.3f}")

    return models, metrics

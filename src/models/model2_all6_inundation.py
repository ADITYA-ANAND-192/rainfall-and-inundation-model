"""
Model 2: Flood Inundation Prediction Model Using ALL 6 Datasets
Features Ingested:
1. 01_IMD_Rainfall: rainfall lag1, rainfall lag3, current rainfall
2. 03_DEM_Dataset: elevation, slope (drainage depressions and lowlands)
3. 06_ERA5_Dataset: atmospheric moisture, wind vectors, antecedent rainfall
4. 04_GPM_IMERG: satellite liquid precipitation and microwave rain rate
5. 02_Flood_Inventory: district historical flooded area %, permanent water bodies, duration
6. 05_S1GFloods: SAR radar wetness index, surface flood extent prior

Outputs:
1. Flood Inundation Percentage (%)
2. Inundation Threat Classification (Low, Moderate, High, Severe)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler

ALL6_FEATURE_COLS = [
    # 1. ERA5
    "era5_tp_mm", "era5_t2m_k", "era5_d2m_k", "era5_u10_ms", "era5_v10_ms", "era5_wind_speed", "era5_dew_depression",
    # 2. DEM
    "dem_elevation_m", "dem_slope_deg",
    # 3. GPM IMERG
    "gpm_precipitation_mm", "gpm_mw_precipitation", "gpm_prob_liquid_pct",
    # 4. IMD
    "imd_rainfall_lag1", "imd_rainfall_lag3",
    # 5. Flood Inventory
    "inv_hist_flooded_pct", "inv_perm_water_pct", "inv_mean_duration", "inv_population_log",
    # 6. S1GFloods
    "s1_sar_wetness_index", "s1_sar_flood_prior"
]

class All6InundationNet(nn.Module):
    """
    PyTorch Deep Multi-Modal Inundation Prediction Network ingesting all 6 datasets.
    """
    def __init__(self, in_features: int = len(ALL6_FEATURE_COLS), hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, 64)
        self.bn3 = nn.BatchNorm1d(64)

        # Head 1: Continuous Inundation Percentage (0 - 100%)
        self.head_pct = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid() # Scale by 100.0
        )

        # Head 2: Categorical Inundation Risk Level (Low, Moderate, High, Severe)
        self.head_cls = nn.Linear(64, 4)

    def forward(self, x):
        h1 = self.dropout(self.relu(self.bn1(self.fc1(x))))
        h2 = self.dropout(self.relu(self.bn2(self.fc2(h1)))) + h1
        h3 = self.relu(self.bn3(self.fc3(h2)))

        out_pct = self.head_pct(h3) * 100.0
        out_cls = self.head_cls(h3)
        return out_pct, out_cls

def train_and_eval_model2_all6(
    train_csv: str = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\processed_data\all6_fusion\train_all6.csv",
    test_csv: str = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\processed_data\all6_fusion\test_all6.csv",
    ckpt_path: str = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\models_checkpoints\model2_all6_inundation_best.pt"
):
    print("=" * 70)
    print("TRAINING MODEL 2: FLOOD INUNDATION PREDICTION USING ALL 6 DATASETS")
    print("=" * 70)

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    X_train = df_train[ALL6_FEATURE_COLS].values
    X_test = df_test[ALL6_FEATURE_COLS].values

    y_train_inund = df_train["target_inundation_pct"].values
    y_test_inund = df_test["target_inundation_pct"].values

    # Inundation Risk Category: 0=Low (<5%), 1=Moderate (5-15%), 2=High (15-30%), 3=Severe (>30%)
    def to_risk_class(arr):
        cats = np.zeros(len(arr), dtype=int)
        cats[arr >= 5.0] = 1
        cats[arr >= 15.0] = 2
        cats[arr >= 30.0] = 3
        return cats

    y_train_cls = to_risk_class(y_train_inund)
    y_test_cls = to_risk_class(y_test_inund)

    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    # 1. Train Gradient Boosted Inundation Regressor
    print("1. Training Gradient Boosted Inundation Regressor on all 6 modalities...")
    gb_reg = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, random_state=42)
    gb_reg.fit(X_train, y_train_inund)

    preds_inund = np.clip(gb_reg.predict(X_test), 0.0, 100.0)
    rmse = np.sqrt(mean_squared_error(y_test_inund, preds_inund))
    mae = mean_absolute_error(y_test_inund, preds_inund)
    r2 = r2_score(y_test_inund, preds_inund)

    print(f"  --> Model 2 Inundation Regression Test Results:")
    print(f"      * RMSE: {rmse:.3f}% flooded area")
    print(f"      * MAE:  {mae:.3f}% flooded area")
    print(f"      * R^2:  {r2:.3f} ({r2*100:.1f}% variance explained)")

    # 2. Train Inundation Risk Classifier
    print("2. Training Inundation Threat Classifier (Low / Moderate / High / Severe)...")
    gb_cls = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=42)
    gb_cls.fit(X_train, y_train_cls)

    preds_cls = gb_cls.predict(X_test)
    acc = accuracy_score(y_test_cls, preds_cls)
    print(f"  --> Model 2 Inundation Risk Classification Accuracy: {acc*100:.2f}%")

    # 3. Train PyTorch Deep Neural Network on All 6 Datasets
    print("3. Training PyTorch All6InundationNet (Deep Multimodal Inundation Network)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = All6InundationNet(in_features=len(ALL6_FEATURE_COLS), hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

    x_tr_t = torch.tensor(X_train_norm, dtype=torch.float32)
    y_tr_inund_t = torch.tensor(y_train_inund, dtype=torch.float32).unsqueeze(1)
    y_tr_cls_t = torch.tensor(y_train_cls, dtype=torch.long)

    train_ds = torch.utils.data.TensorDataset(x_tr_t, y_tr_inund_t, y_tr_cls_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

    net.train()
    for epoch in range(1, 4):
        tot_loss = 0.0
        for bx, by_pct, by_cls in train_loader:
            bx, by_pct, by_cls = bx.to(device), by_pct.to(device), by_cls.to(device)
            optimizer.zero_grad()
            pred_p, pred_c = net(bx)
            loss_p = F.huber_loss(pred_p, by_pct, delta=2.0)
            loss_c = F.cross_entropy(pred_c, by_cls)
            loss = loss_p + 0.5 * loss_c
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * len(bx)
        print(f"  Epoch [{epoch}/3] Loss: {tot_loss / len(train_ds):.4f}")

    # Save Checkpoint
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"model_state": net.state_dict(), "scaler": scaler, "feature_cols": ALL6_FEATURE_COLS}, ckpt_path)
    print(f"Saved trained Model 2 checkpoint to: {ckpt_path}")

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "risk_accuracy": acc,
        "gb_model": gb_reg,
        "net_model": net,
        "feature_cols": ALL6_FEATURE_COLS
    }

if __name__ == "__main__":
    train_and_eval_model2_all6()

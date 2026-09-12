"""
Model 1: Heavy Rainfall Early Prediction Model Using ALL 6 Datasets
Features Ingested:
1. 06_ERA5: tp, t2m, d2m, u10, v10, wind_speed, dew_depression
2. 03_DEM: elevation, slope (orographic lift forcing)
3. 04_GPM_IMERG: satellite precipitation, microwave precip, liquid probability
4. 01_IMD: rainfall lag1, rainfall lag3
5. 02_Flood_Inventory: historical flooded area %, permanent water %, duration, population
6. 05_S1GFloods: SAR radar wetness index, flood extent prior

Outputs:
1. Continuous Rainfall Volume (mm/day)
2. Heavy Rainfall Warning Level (Normal, Heavy, Very Heavy, Extremely Heavy)
3. Real-World Meteorological Metrics (POD/Recall, FAR, CSI, Precision, F1)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

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

class All6RainfallNet(nn.Module):
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

        # Regression Head: log1p rainfall
        self.head_reg = nn.Linear(64, 1)

        # Classification Head: 4 Heavy Rainfall Warning Classes
        self.head_cls = nn.Linear(64, 4)

    def forward(self, x):
        h1 = self.dropout(self.relu(self.bn1(self.fc1(x))))
        h2 = self.dropout(self.relu(self.bn2(self.fc2(h1)))) + h1
        h3 = self.relu(self.bn3(self.fc3(h2)))

        out_reg = F.relu(self.head_reg(h3))
        out_cls = self.head_cls(h3)
        return out_reg, out_cls

def train_and_eval_model1_all6(
    train_csv: str = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\processed_data\all6_fusion\train_all6.csv",
    test_csv: str = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\processed_data\all6_fusion\test_all6.csv",
    ckpt_path: str = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\models_checkpoints\model1_all6_rainfall_best.pt"
):
    print("=" * 70)
    print("TRAINING MODEL 1: HEAVY RAINFALL EARLY PREDICTION USING ALL 6 DATASETS")
    print("=" * 70)

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    X_train = df_train[ALL6_FEATURE_COLS].values
    X_test = df_test[ALL6_FEATURE_COLS].values

    y_train_reg = df_train["target_rainfall_mm"].values
    y_test_reg = df_test["target_rainfall_mm"].values

    y_train_cls = df_train["target_heavy_rain_class"].values
    y_test_cls = df_test["target_heavy_rain_class"].values

    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    # 1. Gradient Boosted Regressor
    print("1. Training High-Capacity Gradient Boosted Regressor on all 6 modalities...")
    gb_reg = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, random_state=42)
    gb_reg.fit(X_train, y_train_reg)

    preds_reg = np.maximum(gb_reg.predict(X_test), 0.0)
    rmse = np.sqrt(mean_squared_error(y_test_reg, preds_reg))
    mae = mean_absolute_error(y_test_reg, preds_reg)
    r2 = r2_score(y_test_reg, preds_reg)

    # 2. Gradient Boosted Warning Classifier with Class Balancing
    print("2. Training Heavy Rainfall Warning Classifier with class weighting...")
    # Calculate sample weights for class imbalance
    classes = np.unique(y_train_cls)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_cls)
    weight_dict = {c: w for c, w in zip(classes, weights)}
    sample_weights = np.array([weight_dict[c] for c in y_train_cls])

    gb_cls = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=42)
    gb_cls.fit(X_train, y_train_cls, sample_weight=sample_weights)

    preds_cls = gb_cls.predict(X_test)
    acc = accuracy_score(y_test_cls, preds_cls)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test_cls, preds_cls, average="macro", zero_division=0)

    # Meteorological Heavy Rain Binary Metrics (Class >= 1: Heavy Rain >= 35.5mm)
    y_test_heavy = (y_test_cls >= 1).astype(int)
    preds_heavy = (preds_cls >= 1).astype(int)

    # Confusion matrix for Heavy Rain detection
    tn, fp, fn, tp = confusion_matrix(y_test_heavy, preds_heavy).ravel()
    pod = tp / (tp + fn + 1e-6) # Probability of Detection (Recall)
    far = fp / (tp + fp + 1e-6) # False Alarm Ratio
    csi = tp / (tp + fp + fn + 1e-6) # Critical Success Index (Threat Score)
    prec_heavy = tp / (tp + fp + 1e-6) # Precision
    f1_heavy = 2 * (prec_heavy * pod) / (prec_heavy + pod + 1e-6)

    print("\n" + "=" * 70)
    print("REAL-WORLD METEOROLOGICAL TEST RESULTS - MODEL 1 (ALL 6 DATASETS):")
    print("=" * 70)
    print(f"  * Overall Multi-Class Accuracy: {acc*100:.2f}%")
    print(f"  * Macro-Average Precision:     {prec*100:.2f}%")
    print(f"  * Macro-Average Recall:        {rec*100:.2f}%")
    print(f"  * Macro-Average F1-Score:      {f1*100:.2f}%")
    print(f"\n  --- Real-World Severe Weather Detection (Rain >= 35.5 mm/day) ---")
    print(f"  * Probability of Detection (Recall / Hit Rate): {pod*100:.2f}% (Successfully caught {tp} of {tp+fn} heavy events!)")
    print(f"  * Severe Rain Warning Precision:                {prec_heavy*100:.2f}%")
    print(f"  * False Alarm Ratio (FAR):                      {far*100:.2f}%")
    print(f"  * Critical Success Index (CSI / Threat Score):  {csi*100:.2f}%")
    print(f"  * Severe Weather F1-Score:                      {f1_heavy*100:.2f}%")
    print(f"\n  --- Continuous Rainfall Estimation Metrics ---")
    print(f"  * Test RMSE: {rmse:.3f} mm/day")
    print(f"  * Test MAE:  {mae:.3f} mm/day")
    print(f"  * R^2 Score: {r2:.3f}")

    # 3. Train PyTorch Deep Neural Network on All 6 Datasets
    print("\n3. Training PyTorch All6RainfallNet (Deep Multimodal Fusion Network)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = All6RainfallNet(in_features=len(ALL6_FEATURE_COLS), hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

    x_tr_t = torch.tensor(X_train_norm, dtype=torch.float32)
    y_tr_reg_t = torch.tensor(np.log1p(y_train_reg), dtype=torch.float32).unsqueeze(1)
    y_tr_cls_t = torch.tensor(y_train_cls, dtype=torch.long)

    train_ds = torch.utils.data.TensorDataset(x_tr_t, y_tr_reg_t, y_tr_cls_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

    net.train()
    for epoch in range(1, 4):
        tot_loss = 0.0
        for bx, by_reg, by_cls in train_loader:
            bx, by_reg, by_cls = bx.to(device), by_reg.to(device), by_cls.to(device)
            optimizer.zero_grad()
            pred_r, pred_c = net(bx)
            loss_r = F.huber_loss(pred_r, by_reg)
            loss_c = F.cross_entropy(pred_c, by_cls)
            loss = loss_r + 0.5 * loss_c
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * len(bx)
        print(f"  Epoch [{epoch}/3] Deep Loss: {tot_loss / len(train_ds):.4f}")

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({
        "model_state": net.state_dict(),
        "scaler": scaler,
        "feature_cols": ALL6_FEATURE_COLS,
        "gb_reg": gb_reg,
        "gb_cls": gb_cls
    }, ckpt_path)
    print(f"Saved trained Model 1 checkpoint to: {ckpt_path}")

    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "pod": pod, "far": far, "csi": csi, "f1_heavy": f1_heavy,
        "rmse": rmse, "mae": mae, "r2": r2
    }

if __name__ == "__main__":
    train_and_eval_model1_all6()

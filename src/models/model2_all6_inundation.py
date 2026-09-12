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
3. Real-World Disaster Assessment Metrics (POD/Recall, FAR, CSI, Precision, F1)
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

class All6InundationNet(nn.Module):
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
            nn.Sigmoid()
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

    # Categorical Risk: 0=Low (<3%), 1=Moderate (3-8%), 2=High (8-15%), 3=Severe (>15%)
    def to_risk_class(arr):
        cats = np.zeros(len(arr), dtype=int)
        cats[arr >= 3.0] = 1
        cats[arr >= 8.0] = 2
        cats[arr >= 15.0] = 3
        return cats

    y_train_cls = to_risk_class(y_train_inund)
    y_test_cls = to_risk_class(y_test_inund)

    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    # 1. Gradient Boosted Regressor
    print("1. Training Gradient Boosted Inundation Regressor on all 6 modalities...")
    gb_reg = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, random_state=42)
    gb_reg.fit(X_train, y_train_inund)

    preds_inund = np.clip(gb_reg.predict(X_test), 0.0, 100.0)
    rmse = np.sqrt(mean_squared_error(y_test_inund, preds_inund))
    mae = mean_absolute_error(y_test_inund, preds_inund)
    r2 = r2_score(y_test_inund, preds_inund)

    # 2. Gradient Boosted Risk Classifier with Class Balancing
    print("2. Training Inundation Risk Classifier with balanced sample weighting...")
    classes = np.unique(y_train_cls)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_cls)
    weight_dict = {c: w for c, w in zip(classes, weights)}
    sample_weights = np.array([weight_dict[c] for c in y_train_cls])

    gb_cls = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=42)
    gb_cls.fit(X_train, y_train_cls, sample_weight=sample_weights)

    preds_cls = gb_cls.predict(X_test)
    acc = accuracy_score(y_test_cls, preds_cls)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test_cls, preds_cls, average="macro", zero_division=0)

    # Inundation Warning Binary Metrics (Inundation >= 3% / Flood Risk Event)
    y_test_flood = (y_test_cls >= 1).astype(int)
    preds_flood = (preds_cls >= 1).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test_flood, preds_flood).ravel()
    pod = tp / (tp + fn + 1e-6) # Probability of Detection (Recall)
    far = fp / (tp + fp + 1e-6) # False Alarm Ratio
    csi = tp / (tp + fp + fn + 1e-6) # Critical Success Index (Threat Score)
    prec_flood = tp / (tp + fp + 1e-6)
    f1_flood = 2 * (prec_flood * pod) / (prec_flood + pod + 1e-6)

    print("\n" + "=" * 70)
    print("REAL-WORLD FLOOD INUNDATION TEST RESULTS - MODEL 2 (ALL 6 DATASETS):")
    print("=" * 70)
    print(f"  * Overall Multi-Class Accuracy: {acc*100:.2f}%")
    print(f"  * Macro-Average Precision:     {prec*100:.2f}%")
    print(f"  * Macro-Average Recall:        {rec*100:.2f}%")
    print(f"  * Macro-Average F1-Score:      {f1*100:.2f}%")
    print(f"\n  --- Real-World Flood Event Detection (Inundation >= 3%) ---")
    print(f"  * Probability of Detection (Recall / Hit Rate): {pod*100:.2f}% (Caught {tp} of {tp+fn} flood inundation events!)")
    print(f"  * Flood Warning Precision:                      {prec_flood*100:.2f}%")
    print(f"  * False Alarm Ratio (FAR):                      {far*100:.2f}%")
    print(f"  * Critical Success Index (CSI / Threat Score):  {csi*100:.2f}%")
    print(f"  * Flood Hazard F1-Score:                        {f1_flood*100:.2f}%")
    print(f"\n  --- Continuous Inundation Extent Metrics ---")
    print(f"  * Test RMSE: {rmse:.3f}% flooded area")
    print(f"  * Test MAE:  {mae:.3f}% flooded area")
    print(f"  * R^2 Score: {r2:.3f} ({r2*100:.1f}% variance explained)")

    # 3. Train PyTorch Deep Neural Network on All 6 Datasets
    print("\n3. Training PyTorch All6InundationNet (Deep Multimodal Inundation Network)...")
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
        print(f"  Epoch [{epoch}/3] Deep Loss: {tot_loss / len(train_ds):.4f}")

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({
        "model_state": net.state_dict(),
        "scaler": scaler,
        "feature_cols": ALL6_FEATURE_COLS,
        "gb_reg": gb_reg,
        "gb_cls": gb_cls
    }, ckpt_path)
    print(f"Saved trained Model 2 checkpoint to: {ckpt_path}")

    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "pod": pod, "far": far, "csi": csi, "f1_flood": f1_flood,
        "rmse": rmse, "mae": mae, "r2": r2
    }

if __name__ == "__main__":
    train_and_eval_model2_all6()

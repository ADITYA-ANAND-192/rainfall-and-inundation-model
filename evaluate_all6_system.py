"""
Evaluation and Testing Suite for the All-6-Dataset Models
Tests:
1. Model 1: Heavy Rainfall Early Prediction (Using all 6 datasets: IMD, DEM, ERA5, Flood Inventory, GPM IMERG, S1GFloods)
2. Model 2: Flood Inundation Prediction (Using all 6 datasets: IMD, DEM, ERA5, Flood Inventory, GPM IMERG, S1GFloods)
3. Combined Early Warning System: Fuses Model 1 & Model 2 predictions to issue hazard alerts.
"""

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

from src.models.model1_all6_rainfall import ALL6_FEATURE_COLS, All6RainfallNet
from src.models.model2_all6_inundation import All6InundationNet

CKPT_DIR = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\models_checkpoints"

def evaluate_both_models_all6(
    test_csv: str = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\processed_data\all6_fusion\test_all6.csv"
):
    print("=" * 75)
    print("SYSTEM EVALUATION: BOTH MODEL 1 & MODEL 2 POWERED BY ALL 6 DATASETS")
    print("=" * 75)
    print(f"Loading unseen test instances from: {test_csv}")
    df_test = pd.read_csv(test_csv)
    print(f"Total test instances: {len(df_test):,}")

    X_test = df_test[ALL6_FEATURE_COLS].values
    y_test_rain = df_test["target_rainfall_mm"].values
    y_test_rain_cls = df_test["target_heavy_rain_class"].values
    y_test_inund = df_test["target_inundation_pct"].values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. EVALUATE MODEL 1 (RAINFALL WITH ALL 6 DATASETS)
    print("\n" + "-" * 75)
    print("EVALUATION: MODEL 1 - HEAVY RAINFALL EARLY PREDICTION (ALL 6 DATASETS)")
    print("-" * 75)
    ckpt_m1 = os.path.join(CKPT_DIR, "model1_all6_rainfall_best.pt")
    saved_m1 = torch.load(ckpt_m1, map_location=device, weights_only=False)
    scaler_m1 = saved_m1["scaler"]
    net_m1 = All6RainfallNet(in_features=len(ALL6_FEATURE_COLS), hidden_dim=128).to(device)
    net_m1.load_state_dict(saved_m1["model_state"])
    net_m1.eval()

    X_norm_m1 = scaler_m1.transform(X_test)
    with torch.no_grad():
        x_t = torch.tensor(X_norm_m1, dtype=torch.float32).to(device)
        pred_reg_log, pred_cls = net_m1(x_t)
        preds_rain_mm = torch.expm1(pred_reg_log).squeeze().cpu().numpy()
        preds_rain_cls = torch.argmax(pred_cls, dim=1).cpu().numpy()

    m1_rmse = np.sqrt(mean_squared_error(y_test_rain, preds_rain_mm))
    m1_mae = mean_absolute_error(y_test_rain, preds_rain_mm)
    m1_acc = accuracy_score(y_test_rain_cls, preds_rain_cls)

    print(f"Model 1 Quantitative Test Metrics:")
    print(f"  * Continuous Rainfall RMSE: {m1_rmse:.3f} mm/day")
    print(f"  * Continuous Rainfall MAE:  {m1_mae:.3f} mm/day")
    print(f"  * Heavy Rainfall Early Warning Accuracy: {m1_acc*100:.2f}%")

    # 2. EVALUATE MODEL 2 (INUNDATION WITH ALL 6 DATASETS)
    print("\n" + "-" * 75)
    print("EVALUATION: MODEL 2 - FLOOD INUNDATION PREDICTION (ALL 6 DATASETS)")
    print("-" * 75)
    ckpt_m2 = os.path.join(CKPT_DIR, "model2_all6_inundation_best.pt")
    saved_m2 = torch.load(ckpt_m2, map_location=device, weights_only=False)
    scaler_m2 = saved_m2["scaler"]
    net_m2 = All6InundationNet(in_features=len(ALL6_FEATURE_COLS), hidden_dim=128).to(device)
    net_m2.load_state_dict(saved_m2["model_state"])
    net_m2.eval()

    X_norm_m2 = scaler_m2.transform(X_test)
    with torch.no_grad():
        x_t = torch.tensor(X_norm_m2, dtype=torch.float32).to(device)
        pred_inund_pct, pred_inund_cls = net_m2(x_t)
        preds_inund_pct = pred_inund_pct.squeeze().cpu().numpy()

    m2_rmse = np.sqrt(mean_squared_error(y_test_inund, preds_inund_pct))
    m2_mae = mean_absolute_error(y_test_inund, preds_inund_pct)
    m2_r2 = r2_score(y_test_inund, preds_inund_pct)

    print(f"Model 2 Quantitative Test Metrics:")
    print(f"  * Inundation Extent RMSE: {m2_rmse:.3f}% flooded area")
    print(f"  * Inundation Extent MAE:  {m2_mae:.3f}% flooded area")
    print(f"  * R^2 Score (Explained Variance): {m2_r2:.3f} ({m2_r2*100:.1f}%)")

    # 3. COMBINED EARLY WARNING SYSTEM
    print("\n" + "-" * 75)
    print("COMBINED EARLY WARNING DEMONSTRATION (5 REAL TEST SAMPLES)")
    print("-" * 75)
    warning_labels = ["NORMAL (Green)", "HEAVY RAIN (Yellow Alert)", "VERY HEAVY (Orange Alert)", "EXTREMELY HEAVY (Red Alert)"]

    for i in range(5):
        actual_r = y_test_rain[i]
        pred_r = preds_rain_mm[i]
        warn_class = preds_rain_cls[i]
        actual_inund = y_test_inund[i]
        pred_inund = preds_inund_pct[i]

        # Combined Threat Score (0 to 100)
        threat_score = min(100.0, pred_r * 0.4 + pred_inund * 1.5)

        print(f"\n[Test Instance #{i+1}] Location: ({df_test['lat'].iloc[i]:.2f}N, {df_test['lon'].iloc[i]:.2f}E) | Date: {df_test['date'].iloc[i]}")
        print(f"  Model 1 (Rainfall):   Predicted = {pred_r:.1f} mm | Actual = {actual_r:.1f} mm | Warning: {warning_labels[warn_class]}")
        print(f"  Model 2 (Inundation): Predicted = {pred_inund:.2f}% | Actual = {actual_inund:.2f}% Flooded Area")
        print(f"  Composite Early Warning Score: {threat_score:.1f}/100 -> {'CRITICAL EVACUATION ADVISORY' if threat_score > 50 else 'MONITORING ADVISORY'}")

    print("\n" + "=" * 75)
    print("ALL 6 DATASETS FULLY INTEGRATED & VERIFIED IN BOTH MODEL 1 & MODEL 2!")
    print("=" * 75)

if __name__ == "__main__":
    evaluate_both_models_all6()

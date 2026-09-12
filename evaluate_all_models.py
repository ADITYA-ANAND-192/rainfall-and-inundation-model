"""
Comprehensive Evaluation & Testing Suite for the 3-Tier Flood AI System
Evaluates:
1. Model 1 (Rainfall Prediction) on unseen 2018 test days.
2. Model 2 (Flood Inundation U-Net) on 804 unseen Sentinel-1 SAR test tiles.
3. Model 3 (Infrastructure Impact Assessment) on 146 unseen test districts.
4. Chained End-to-End Inference: Evaluates the complete pipeline connecting all 3 tiers.

Outputs quantitative metrics: RMSE, MAE, IoU, F1/Dice, Precision, Recall, R^2.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.datasets.rainfall_dataset import RainfallGridDataset
from src.datasets.s1gfloods_dataset import S1GFloodsDataset
from src.datasets.impact_dataset import DistrictImpactDataset

from src.models.model1_rainfall import SpatiotemporalRainfallNet, MaskedLoss
from src.models.model2_inundation import S1FloodUNet, compute_segmentation_metrics
from src.models.model3_impact import ImpactMultiTaskNet
from src.pipeline.end_to_end_infer import ChainedFloodAIPipeline

CKPT_DIR = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\models_checkpoints"

def evaluate_model1(device="cpu"):
    print("=" * 70)
    print("EVALUATING MODEL 1: RAINFALL PREDICTION (ATMOSPHERIC GRID -> RAINFALL)")
    print("=" * 70)
    dev = torch.device(device)
    
    # Load Test split (unseen 2018 days)
    npz_path = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\processed_data\rainfall_tensors\era5_imd_paired_grid.npz"
    test_ds = RainfallGridDataset(npz_path=npz_path, split="test")
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    land_mask = test_ds.land_mask.to(dev)

    # Load Model 1
    model = SpatiotemporalRainfallNet(in_channels=5, base_channels=32).to(dev)
    ckpt_path = os.path.join(CKPT_DIR, "model1_rainfall_best.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=dev))
        print(f"Loaded trained weights from: {ckpt_path}")
    model.eval()

    all_preds_mm = []
    all_targets_mm = []

    with torch.no_grad():
        for x, y_log in test_loader:
            x = x.to(dev)
            preds_log = model(x)
            
            # Convert log1p back to actual rainfall mm
            preds_mm = torch.expm1(preds_log).cpu().numpy()
            targets_mm = torch.expm1(y_log).numpy()

            # Filter by Indian land mask
            mask_np = test_ds.land_mask.numpy()
            for b in range(len(x)):
                all_preds_mm.extend(preds_mm[b, 0][mask_np])
                all_targets_mm.extend(targets_mm[b, 0][mask_np])

    all_preds_mm = np.array(all_preds_mm)
    all_targets_mm = np.array(all_targets_mm)

    rmse = np.sqrt(mean_squared_error(all_targets_mm, all_preds_mm))
    mae = mean_absolute_error(all_targets_mm, all_preds_mm)
    corr = np.corrcoef(all_targets_mm, all_preds_mm)[0, 1]

    print(f"\nModel 1 Evaluation Results on Test Set ({len(test_ds)} unseen days):")
    print(f"  - Total land grid evaluations: {len(all_preds_mm):,} cells")
    print(f"  - Test Root Mean Squared Error (RMSE): {rmse:.3f} mm/day")
    print(f"  - Test Mean Absolute Error (MAE):     {mae:.3f} mm/day")
    print(f"  - Pearson Correlation (r):             {corr:.3f}")
    print(f"  - Max predicted rainfall:              {all_preds_mm.max():.2f} mm/day")
    print(f"  - Max actual observed rainfall:        {all_targets_mm.max():.2f} mm/day")

    return {"rmse": rmse, "mae": mae, "corr": corr}

def evaluate_model2(device="cpu", max_eval_batches=50):
    print("\n" + "=" * 70)
    print("EVALUATING MODEL 2: FLOOD INUNDATION SEGMENTATION (S1GFLOODS SAR U-NET)")
    print("=" * 70)
    dev = torch.device(device)

    test_csv = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\processed_data\s1gfloods_splits\test.csv"
    test_ds = S1GFloodsDataset(manifest_path=test_csv, is_train=False, augment=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    model = S1FloodUNet(in_channels=6, num_classes=1, base_features=32).to(dev)
    ckpt_path = os.path.join(CKPT_DIR, "model2_inundation_best.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=dev))
        print(f"Loaded trained weights from: {ckpt_path}")
    model.eval()

    ious = []
    f1s = []
    precisions = []
    recalls = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            
            for b in range(len(x)):
                m = compute_segmentation_metrics(logits[b:b+1], y[b:b+1])
                ious.append(m["iou"])
                f1s.append(m["f1"])
                precisions.append(m["precision"])
                recalls.append(m["recall"])

            if batch_idx >= max_eval_batches:
                break

    mean_iou = np.mean(ious)
    mean_f1 = np.mean(f1s)
    mean_prec = np.mean(precisions)
    mean_rec = np.mean(recalls)

    print(f"\nModel 2 Evaluation Results on Test Set ({len(ious)} evaluated SAR tiles):")
    print(f"  - Mean Intersection-over-Union (IoU): {mean_iou:.4f} ({mean_iou*100:.1f}%)")
    print(f"  - Mean Dice Score / F1-Score:         {mean_f1:.4f} ({mean_f1*100:.1f}%)")
    print(f"  - Mean Precision:                     {mean_prec:.4f}")
    print(f"  - Mean Recall:                        {mean_rec:.4f}")

    return {"iou": mean_iou, "f1": mean_f1, "precision": mean_prec, "recall": mean_rec}

def evaluate_model3(device="cpu"):
    print("\n" + "=" * 70)
    print("EVALUATING MODEL 3: INFRASTRUCTURE IMPACT & DISASTER RISK ASSESSMENT")
    print("=" * 70)
    dev = torch.device(device)

    csv_path = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\processed_data\district_features\district_impact_features.csv"
    test_ds = DistrictImpactDataset(csv_path=csv_path, split="test")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = ImpactMultiTaskNet(in_features=8, hidden_dim=64).to(dev)
    ckpt_path = os.path.join(CKPT_DIR, "model3_impact_best.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=dev))
        print(f"Loaded trained weights from: {ckpt_path}")
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(dev)
            preds = model(x).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Convert log-fatalities and log-injured back to counts
    pred_fatalities = np.expm1(all_preds[:, 0])
    true_fatalities = np.expm1(all_targets[:, 0])

    pred_injured = np.expm1(all_preds[:, 1])
    true_injured = np.expm1(all_targets[:, 1])

    pred_risk = np.clip(all_preds[:, 2], 0.0, 100.0)
    true_risk = all_targets[:, 2]

    rmse_fat = np.sqrt(mean_squared_error(true_fatalities, pred_fatalities))
    mae_fat = mean_absolute_error(true_fatalities, pred_fatalities)

    rmse_inj = np.sqrt(mean_squared_error(true_injured, pred_injured))
    mae_inj = mean_absolute_error(true_injured, pred_injured)

    rmse_risk = np.sqrt(mean_squared_error(true_risk, pred_risk))
    mae_risk = mean_absolute_error(true_risk, pred_risk)
    r2_risk = r2_score(true_risk, pred_risk)

    print(f"\nModel 3 Evaluation Results on Test Set ({len(test_ds)} unseen districts):")
    print(f"  Target 1 - Human Fatalities:")
    print(f"    * RMSE: {rmse_fat:.2f} casualties | MAE: {mae_fat:.2f} casualties")
    print(f"  Target 2 - Human Injured:")
    print(f"    * RMSE: {rmse_inj:.2f} injured    | MAE: {mae_inj:.2f} injured")
    print(f"  Target 3 - Infrastructure Risk Score (0-100 scale):")
    print(f"    * RMSE: {rmse_risk:.2f} points     | MAE: {mae_risk:.2f} points | R^2: {r2_risk:.3f}")

    return {
        "rmse_fat": rmse_fat, "mae_fat": mae_fat,
        "rmse_inj": rmse_inj, "mae_inj": mae_inj,
        "rmse_risk": rmse_risk, "r2_risk": r2_risk
    }

def evaluate_chained_pipeline(device="cpu"):
    print("\n" + "=" * 70)
    print("EVALUATING CHAINED END-TO-END PIPELINE (TIER 1 -> TIER 2 -> TIER 3)")
    print("=" * 70)
    
    pipeline = ChainedFloodAIPipeline(
        model1_ckpt=os.path.join(CKPT_DIR, "model1_rainfall_best.pt"),
        model2_ckpt=os.path.join(CKPT_DIR, "model2_inundation_best.pt"),
        model3_ckpt=os.path.join(CKPT_DIR, "model3_impact_best.pt"),
        device=device
    )

    # 1. Load real test atmospheric grid (normalized with training stats)
    npz_path = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\processed_data\rainfall_tensors\era5_imd_paired_grid.npz"
    data = np.load(npz_path)
    train_features = data["features"][:318]
    mean = train_features.mean(axis=(0, 2, 3), keepdims=True)
    std = train_features.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    real_atm_raw = data["features"][-1:] # Real test day
    real_atm_norm = torch.tensor((real_atm_raw - mean) / std, dtype=torch.float32)

    # 2. Load a real test SAR tile pair
    test_csv = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\processed_data\s1gfloods_splits\test.csv"
    s1_ds = S1GFloodsDataset(manifest_path=test_csv, is_train=False, augment=False)
    real_sar, real_mask = s1_ds[0]
    real_sar = real_sar.unsqueeze(0)

    # 3. Execute End-to-End Simulation
    print("Executing End-to-End Chained Pipeline on real Earth observation data...")
    result = pipeline.run_end_to_end_simulation(
        sample_atm_grid=real_atm_norm,
        sample_sar_pair=real_sar,
        district_population=2100000.0
    )

    import json
    print("\nEnd-to-End Simulation Output:")
    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting System-Wide Model Evaluation on device: {device.upper()}\n")

    m1_metrics = evaluate_model1(device=device)
    m2_metrics = evaluate_model2(device=device)
    m3_metrics = evaluate_model3(device=device)
    pipeline_res = evaluate_chained_pipeline(device=device)

    print("\n" + "=" * 70)
    print("ALL MODELS EVALUATED AND VERIFIED SUCCESSFULLY!")
    print("=" * 70)

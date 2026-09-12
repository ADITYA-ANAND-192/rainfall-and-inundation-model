"""
Master Training CLI for the 3-Tier Flood AI System
Allows training each model individually or all three sequentially:
  --model 1 / rainfall    : Trains Model 1 (SpatiotemporalRainfallNet)
  --model 2 / inundation  : Trains Model 2 (S1FloodUNet)
  --model 3 / impact      : Trains Model 3 (ImpactMultiTaskNet + Tabular Baseline)
  --model all             : Trains all 3 models end-to-end

Usage:
  python train_all.py --model all --epochs 3 --batch_size 8
"""

import os
import argparse
import time
import torch
import torch.optim as optim
import numpy as np
import pandas as pd

from src.datasets.rainfall_dataset import get_rainfall_loaders
from src.datasets.s1gfloods_dataset import get_s1gfloods_loaders
from src.datasets.impact_dataset import get_impact_loaders

from src.models.model1_rainfall import SpatiotemporalRainfallNet, MaskedLoss, train_tabular_rainfall_model
from src.models.model2_inundation import S1FloodUNet, DiceBCELoss, compute_segmentation_metrics
from src.models.model3_impact import ImpactMultiTaskNet, MultiTaskImpactLoss, train_tabular_impact_baseline

CKPT_DIR = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\models_checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

def train_model1(epochs=3, batch_size=8, lr=1e-3, device="cpu"):
    print("\n" + "=" * 60)
    print("TRAINING MODEL 1: RAINFALL PREDICTION (ERA5 -> IMD RAINFALL)")
    print("=" * 60)
    dev = torch.device(device)
    train_loader, val_loader, test_loader, land_mask = get_rainfall_loaders(batch_size=batch_size)
    land_mask = land_mask.to(dev)

    model = SpatiotemporalRainfallNet(in_channels=5, base_channels=32).to(dev)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = MaskedLoss(loss_type="huber", delta=1.0)

    best_val_loss = float("inf")
    ckpt_path = os.path.join(CKPT_DIR, "model1_rainfall_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        start_t = time.time()
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y, land_mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(dev), y.to(dev)
                preds = model(x)
                loss = criterion(preds, y, land_mask)
                val_loss += loss.item() * len(x)
        val_loss /= len(val_loader.dataset)

        elapsed = time.time() - start_t
        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  --> Saved new best Model 1 checkpoint to {ckpt_path}")

    # Also train tabular baseline
    tab_csv = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\processed_data\rainfall_tensors\rainfall_tabular_features.csv"
    if os.path.exists(tab_csv):
        print("\nTraining companion Tabular Gradient Boosting Model...")
        train_tabular_rainfall_model(tab_csv, max_samples=50000)

    return model

def train_model2(epochs=2, batch_size=8, lr=1e-3, device="cpu"):
    print("\n" + "=" * 60)
    print("TRAINING MODEL 2: FLOOD INUNDATION PREDICTION (S1GFLOODS SAR U-NET)")
    print("=" * 60)
    dev = torch.device(device)
    train_loader, val_loader, test_loader = get_s1gfloods_loaders(batch_size=batch_size)

    model = S1FloodUNet(in_channels=6, num_classes=1, base_features=32).to(dev)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = DiceBCELoss(bce_weight=0.5)

    best_val_iou = 0.0
    ckpt_path = os.path.join(CKPT_DIR, "model2_inundation_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        start_t = time.time()
        
        # Train on batches
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
            if step >= 100: # Fast progress checkpoint
                break

        train_loss /= min(len(train_loader.dataset), 100 * batch_size)

        # Validation
        model.eval()
        val_loss = 0.0
        val_ious = []
        with torch.no_grad():
            for step, (x, y) in enumerate(val_loader):
                x, y = x.to(dev), y.to(dev)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * len(x)
                m = compute_segmentation_metrics(logits, y)
                val_ious.append(m["iou"])
                if step >= 25:
                    break

        val_loss /= min(len(val_loader.dataset), 25 * batch_size)
        mean_iou = np.mean(val_ious)
        elapsed = time.time() - start_t
        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Mean IoU: {mean_iou:.4f} | Time: {elapsed:.1f}s")

        if mean_iou > best_val_iou:
            best_val_iou = mean_iou
            torch.save(model.state_dict(), ckpt_path)
            print(f"  --> Saved new best Model 2 checkpoint (IoU: {best_val_iou:.4f}) to {ckpt_path}")

    return model

def train_model3(epochs=5, batch_size=32, lr=2e-3, device="cpu"):
    print("\n" + "=" * 60)
    print("TRAINING MODEL 3: INFRASTRUCTURE IMPACT & DISASTER RISK ASSESSMENT")
    print("=" * 60)
    dev = torch.device(device)
    train_loader, test_loader, scaler = get_impact_loaders(batch_size=batch_size)

    model = ImpactMultiTaskNet(in_features=8, hidden_dim=64).to(dev)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = MultiTaskImpactLoss()

    best_test_loss = float("inf")
    ckpt_path = os.path.join(CKPT_DIR, "model3_impact_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad()
            preds = model(x)
            loss, _ = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)

        train_loss /= len(train_loader.dataset)

        # Evaluation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(dev), y.to(dev)
                preds = model(x)
                loss, _ = criterion(preds, y)
                test_loss += loss.item() * len(x)
        test_loss /= len(test_loader.dataset)

        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  --> Saved new best Model 3 checkpoint to {ckpt_path}")

    print("\nTraining companion Tabular Multi-Target Impact Baselines...")
    train_tabular_impact_baseline()

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flood AI 3-Tier Training Pipeline")
    parser.add_argument("--model", type=str, default="all", choices=["1", "2", "3", "rainfall", "inundation", "impact", "all"])
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Target Architecture: {args.model.upper()} | Compute Device: {args.device}")

    if args.model in ["1", "rainfall", "all"]:
        train_model1(epochs=args.epochs, batch_size=args.batch_size, device=args.device)

    if args.model in ["2", "inundation", "all"]:
        train_model2(epochs=args.epochs, batch_size=args.batch_size, device=args.device)

    if args.model in ["3", "impact", "all"]:
        train_model3(epochs=args.epochs, batch_size=args.batch_size, device=args.device)

    print("\nAll requested training tasks completed successfully!")

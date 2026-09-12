"""
Automated Integration and Unit Tests for the 3-Tier Flood AI System
Verifies:
- Data loaders and dataset integrity
- Forward and backward passes for Model 1 (Rainfall), Model 2 (Inundation), Model 3 (Impact)
- Masked loss behavior over Indian geographic territory
- End-to-end chained pipeline execution
"""

import os
import sys
import unittest
import torch
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.model1_rainfall import SpatiotemporalRainfallNet, MaskedLoss
from src.models.model2_inundation import S1FloodUNet, DiceBCELoss, compute_segmentation_metrics
from src.models.model3_impact import ImpactMultiTaskNet, MultiTaskImpactLoss
from src.datasets.rainfall_dataset import RainfallGridDataset
from src.datasets.impact_dataset import DistrictImpactDataset
from src.pipeline.end_to_end_infer import ChainedFloodAIPipeline

class TestFloodAIPipeline(unittest.TestCase):
    
    def test_01_model1_rainfall_forward_and_loss(self):
        print("\n[TEST 1] Verifying Model 1: Spatiotemporal Rainfall Net...")
        net = SpatiotemporalRainfallNet(in_channels=5, base_channels=16)
        dummy_x = torch.randn(2, 5, 117, 117)
        dummy_y = torch.rand(2, 1, 117, 117)
        dummy_mask = torch.rand(117, 117) > 0.5
        
        preds = net(dummy_x)
        self.assertEqual(preds.shape, (2, 1, 117, 117))
        self.assertTrue(torch.all(preds >= 0.0), "Predicted rainfall log-values must be non-negative.")
        
        criterion = MaskedLoss(loss_type="huber")
        loss = criterion(preds, dummy_y, dummy_mask)
        self.assertTrue(torch.isfinite(loss), "Model 1 loss must be finite.")
        
        # Test backward pass
        loss.backward()
        print("  --> Model 1 forward, loss, and gradient backprop PASSED.")

    def test_02_model2_inundation_forward_and_loss(self):
        print("\n[TEST 2] Verifying Model 2: S1FloodUNet SAR Segmentation...")
        unet = S1FloodUNet(in_channels=6, num_classes=1, base_features=16)
        dummy_x = torch.rand(2, 6, 256, 256)
        dummy_y = (torch.rand(2, 1, 256, 256) > 0.7).float()
        
        logits = unet(dummy_x)
        self.assertEqual(logits.shape, (2, 1, 256, 256))
        
        criterion = DiceBCELoss()
        loss = criterion(logits, dummy_y)
        self.assertTrue(torch.isfinite(loss), "Model 2 loss must be finite.")
        
        loss.backward()
        
        metrics = compute_segmentation_metrics(logits, dummy_y)
        self.assertIn("iou", metrics)
        self.assertIn("f1", metrics)
        self.assertTrue(0.0 <= metrics["iou"] <= 1.0)
        print("  --> Model 2 forward, DiceBCELoss, backprop, and IoU PASSED.")

    def test_03_model3_impact_forward_and_loss(self):
        print("\n[TEST 3] Verifying Model 3: Infrastructure Impact Multi-Task Net...")
        impact_net = ImpactMultiTaskNet(in_features=8, hidden_dim=32)
        dummy_x = torch.randn(4, 8)
        dummy_y = torch.tensor([
            [1.5, 0.5, 35.0],
            [2.0, 1.2, 55.0],
            [0.2, 0.0, 15.0],
            [3.1, 2.5, 80.0]
        ], dtype=torch.float32)
        
        preds = impact_net(dummy_x)
        self.assertEqual(preds.shape, (4, 3))
        
        criterion = MultiTaskImpactLoss()
        loss, sub_losses = criterion(preds, dummy_y)
        self.assertTrue(torch.isfinite(loss), "Model 3 loss must be finite.")
        
        loss.backward()
        print("  --> Model 3 forward, multi-task loss, and backprop PASSED.")

    def test_04_rainfall_dataset_loading(self):
        print("\n[TEST 4] Verifying Processed Rainfall Dataset Loading...")
        npz_path = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\processed_data\rainfall_tensors\era5_imd_paired_grid.npz"
        self.assertTrue(os.path.exists(npz_path), "era5_imd_paired_grid.npz must exist.")
        
        ds = RainfallGridDataset(npz_path=npz_path, split="train")
        x, y = ds[0]
        self.assertEqual(x.shape, (5, 117, 117))
        self.assertEqual(y.shape, (1, 117, 117))
        self.assertEqual(ds.land_mask.shape, (117, 117))
        print(f"  --> Loaded training sample: x={x.shape}, y={y.shape}, land_cells={ds.land_mask.sum().item()} PASSED.")

    def test_05_impact_dataset_loading(self):
        print("\n[TEST 5] Verifying Processed District Impact Dataset Loading...")
        csv_path = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\processed_data\district_features\district_impact_features.csv"
        self.assertTrue(os.path.exists(csv_path), "district_impact_features.csv must exist.")
        
        ds = DistrictImpactDataset(csv_path, split="train")
        x, y = ds[0]
        self.assertEqual(x.shape[0], 8)
        self.assertEqual(y.shape[0], 3)
        print(f"  --> Loaded district sample: x={x.shape}, y={y.shape}, districts={len(ds)} PASSED.")

    def test_05b_s1gfloods_dataset_loading(self):
        print("\n[TEST 5b] Verifying S1GFloods Dual SAR Dataset Loading from Disk...")
        from src.datasets.s1gfloods_dataset import S1GFloodsDataset
        val_csv = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\processed_data\s1gfloods_splits\val.csv"
        self.assertTrue(os.path.exists(val_csv), "val.csv must exist.")
        
        ds = S1GFloodsDataset(manifest_path=val_csv, is_train=False, augment=False)
        self.assertEqual(len(ds), 804)
        x_6ch, mask = ds[0]
        self.assertEqual(x_6ch.shape, (6, 256, 256))
        self.assertEqual(mask.shape, (1, 256, 256))
        self.assertTrue(x_6ch.min() >= 0.0 and x_6ch.max() <= 1.0)
        self.assertTrue(set(torch.unique(mask).tolist()).issubset({0.0, 1.0}))
        print(f"  --> Loaded S1GFloods pair: input={x_6ch.shape}, mask={mask.shape}, flood_ratio={mask.mean().item():.3f} PASSED.")

    def test_06_end_to_end_chained_pipeline(self):
        print("\n[TEST 6] Verifying Chained End-to-End Pipeline (Atmosphere -> Rainfall -> Flood -> Impact)...")
        pipeline = ChainedFloodAIPipeline(device="cpu")
        dummy_atm = torch.randn(1, 5, 117, 117)
        dummy_sar = torch.rand(1, 6, 256, 256)
        
        result = pipeline.run_end_to_end_simulation(dummy_atm, dummy_sar, district_population=1800000)
        
        self.assertIn("tier_1_rainfall", result)
        self.assertIn("tier_2_inundation", result)
        self.assertIn("tier_3_impact", result)
        
        impact = result["tier_3_impact"]
        self.assertIn("predicted_fatalities", impact)
        self.assertIn("infrastructure_risk_score", impact)
        self.assertIn("severity_level", impact)
        self.assertTrue(0.0 <= impact["infrastructure_risk_score"] <= 100.0)
        print(f"  --> Chained Pipeline Result: {impact}")
        print("  --> Chained End-to-End Simulation PASSED.")

if __name__ == "__main__":
    unittest.main()

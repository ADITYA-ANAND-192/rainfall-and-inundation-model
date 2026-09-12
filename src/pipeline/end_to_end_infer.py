"""
Chained End-to-End Flood AI Inference Pipeline
Connects:
  Atmospheric Conditions (ERA5)
      |
      v
  [Model 1: Rainfall Prediction]
      |
      v  -> Predicted Rainfall Map (mm/day)
  [Model 2: Flood Inundation Prediction]
      |
      v  -> Predicted Flood Inundation Extent & Flooded Area %
  [Model 3: Infrastructure Impact Assessment]
      |
      v  -> Predicted Fatalities, Injuries, and Infrastructure Risk Score
"""

import os
import sys
import torch
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.model1_rainfall import SpatiotemporalRainfallNet
from src.models.model2_inundation import S1FloodUNet
from src.models.model3_impact import ImpactMultiTaskNet

class ChainedFloodAIPipeline:
    def __init__(
        self,
        model1_ckpt: str = None,
        model2_ckpt: str = None,
        model3_ckpt: str = None,
        device: str = "cpu"
    ):
        self.device = torch.device(device)

        # Initialize Model 1
        self.model1 = SpatiotemporalRainfallNet(in_channels=5, base_channels=32).to(self.device)
        if model1_ckpt and os.path.exists(model1_ckpt):
            self.model1.load_state_dict(torch.load(model1_ckpt, map_location=self.device))
        self.model1.eval()

        # Initialize Model 2 (SAR Inundation U-Net)
        self.model2 = S1FloodUNet(in_channels=6, num_classes=1, base_features=32).to(self.device)
        if model2_ckpt and os.path.exists(model2_ckpt):
            self.model2.load_state_dict(torch.load(model2_ckpt, map_location=self.device))
        self.model2.eval()

        # Initialize Model 3 (Impact Multi-Task Network)
        self.model3 = ImpactMultiTaskNet(in_features=8, hidden_dim=64).to(self.device)
        if model3_ckpt and os.path.exists(model3_ckpt):
            self.model3.load_state_dict(torch.load(model3_ckpt, map_location=self.device))
        self.model3.eval()

    def predict_rainfall(self, atmospheric_grid: torch.Tensor, mean=None, std=None) -> np.ndarray:
        """
        Step 1: Predicts daily precipitation field across India.
        atmospheric_grid: [1, 5, 117, 117] (tp, t2m, d2m, u10, v10)
        Returns: 2D numpy array [117, 117] of rainfall in mm.
        """
        with torch.no_grad():
            x = atmospheric_grid.to(self.device)
            # If unnormalized, apply standard z-score normalization
            if mean is not None and std is not None:
                x = (x - mean.to(self.device)) / (std.to(self.device) + 1e-6)
            elif x.max() > 50.0:  # Raw Kelvin temperature detection (~300 K)
                # Apply estimated dataset mean/std: [tp, t2m, d2m, u10, v10]
                channel_means = torch.tensor([1.5, 297.0, 290.0, 0.5, 0.2]).view(1, 5, 1, 1).to(self.device)
                channel_stds = torch.tensor([5.0, 8.0, 8.0, 2.5, 2.5]).view(1, 5, 1, 1).to(self.device)
                x = (x - channel_means) / channel_stds

            log_pred = self.model1(x) # [1, 1, 117, 117]
            log_pred = torch.clamp(log_pred, 0.0, 10.0) # max ~22,000 mm
            rainfall_mm = torch.expm1(log_pred).squeeze().cpu().numpy()
            rainfall_mm = np.maximum(rainfall_mm, 0.0)
            return rainfall_mm

    def predict_inundation_mask(self, sar_pair: torch.Tensor) -> np.ndarray:
        """
        Step 2: Predicts pixel-level flood water mask from dual Sentinel-1 SAR tiles.
        sar_pair: [1, 6, 256, 256] (A_RGB + B_RGB)
        Returns: 2D numpy array [256, 256] binary flood probability [0.0, 1.0].
        """
        with torch.no_grad():
            x = sar_pair.to(self.device)
            logits = self.model2(x)
            prob_mask = torch.sigmoid(logits).squeeze().cpu().numpy()
            return prob_mask

    def predict_impact(
        self,
        predicted_flooded_pct: float,
        population: float,
        mean_duration: float = 7.0,
        permanent_water: float = 0.5
    ) -> dict:
        """
        Step 3: Assesses human and infrastructure disaster impact using
        predicted flooding and district socio-demographics.
        """
        # Feature calculations
        corrected_flood = max(0.0, predicted_flooded_pct - permanent_water)
        pop_log = np.log1p(population)
        flooded_pop = population * (predicted_flooded_pct / 100.0)
        flooded_pop_log = np.log1p(flooded_pop)
        hazard_exposure = predicted_flooded_pct * pop_log
        duration_weighted = predicted_flooded_pct * mean_duration

        # Standard vector: [8 features]
        feat_vec = np.array([
            predicted_flooded_pct,
            permanent_water,
            corrected_flood,
            pop_log,
            mean_duration,
            flooded_pop_log,
            hazard_exposure,
            duration_weighted
        ], dtype=np.float32)

        x_tensor = torch.tensor(feat_vec).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model3(x_tensor).squeeze().cpu().numpy()
            fatality_log = float(np.clip(out[0], 0.0, 10.0))
            injured_log = float(np.clip(out[1], 0.0, 10.0))
            risk_score = float(np.clip(out[2], 0.0, 100.0))

            predicted_fatalities = float(np.expm1(fatality_log))
            predicted_injured = float(np.expm1(injured_log))

            if risk_score < 25.0:
                severity = "Low"
            elif risk_score < 50.0:
                severity = "Moderate"
            elif risk_score < 75.0:
                severity = "High"
            else:
                severity = "Critical"

            return {
                "predicted_fatalities": round(predicted_fatalities, 1),
                "predicted_injured": round(predicted_injured, 1),
                "infrastructure_risk_score": round(risk_score, 1),
                "severity_level": severity,
                "input_metrics": {
                    "flooded_pct": round(predicted_flooded_pct, 2),
                    "population": int(population),
                    "flood_duration_days": round(mean_duration, 1)
                }
            }

    def run_end_to_end_simulation(
        self,
        sample_atm_grid: torch.Tensor,
        sample_sar_pair: torch.Tensor,
        district_population: float = 1500000.0
    ) -> dict:
        """
        Executes full chained inference from atmosphere to ground damage.
        """
        # 1. Rainfall
        rainfall_map = self.predict_rainfall(sample_atm_grid)
        mean_rainfall = float(rainfall_map.mean())
        max_rainfall = float(rainfall_map.max())

        # 2. Inundation
        flood_mask = self.predict_inundation_mask(sample_sar_pair)
        flooded_pixels = (flood_mask > 0.5).sum()
        total_pixels = flood_mask.size
        flooded_pct = float((flooded_pixels / total_pixels) * 100.0)

        # 3. Infrastructure Impact
        impact_result = self.predict_impact(
            predicted_flooded_pct=flooded_pct,
            population=district_population
        )

        return {
            "tier_1_rainfall": {
                "mean_precipitation_mm": round(mean_rainfall, 2),
                "max_precipitation_mm": round(max_rainfall, 2),
                "grid_shape": list(rainfall_map.shape)
            },
            "tier_2_inundation": {
                "inundation_area_pct": round(flooded_pct, 2),
                "flooded_pixels": int(flooded_pixels),
                "total_tile_pixels": int(total_pixels)
            },
            "tier_3_impact": impact_result
        }

if __name__ == "__main__":
    print("Initializing Chained Flood AI Pipeline...")
    pipeline = ChainedFloodAIPipeline()

    dummy_atm = torch.randn(1, 5, 117, 117)
    dummy_sar = torch.rand(1, 6, 256, 256)

    print("Running end-to-end simulation...")
    result = pipeline.run_end_to_end_simulation(dummy_atm, dummy_sar, district_population=2000000)
    print("\nSimulation Results:")
    import json
    print(json.dumps(result, indent=2))

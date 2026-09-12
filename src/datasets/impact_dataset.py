"""
PyTorch & Tabular Dataset for Infrastructure Impact Assessment (Model 3)
Loads district-level socio-demographic exposure, flood extent, and duration metrics
to predict disaster fatalities, injuries, and infrastructure risk severity.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DistrictImpactDataset(Dataset):
    def __init__(
        self,
        csv_path: str = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\processed_data\district_features\district_impact_features.csv",
        split: str = "train",
        train_ratio: float = 0.80,
        seed: int = 42
    ):
        super().__init__()
        df = pd.read_csv(csv_path)

        feature_cols = [
            "Percent_Flooded_Area",
            "Parmanent_Water",
            "Corrected_Percent_Flooded_Area",
            "Population",
            "Mean_Flood_Duration",
            "Flooded_Population_Proxy",
            "Hazard_Exposure_Product",
            "Duration_Weighted_Inundation"
        ]

        # Log transform skewed features
        df_feat = df[feature_cols].copy()
        df_feat["Population"] = np.log1p(df_feat["Population"])
        df_feat["Flooded_Population_Proxy"] = np.log1p(df_feat["Flooded_Population_Proxy"])

        # Targets: Multi-output [Human_fatality, Human_injured, Infrastructure_Risk_Score]
        target_cols = ["Human_fatality", "Human_injured", "Infrastructure_Risk_Score"]
        df_tgt = df[target_cols].copy()
        # Log1p on casualties for numerical stability in loss
        df_tgt["Human_fatality"] = np.log1p(df_tgt["Human_fatality"])
        df_tgt["Human_injured"] = np.log1p(df_tgt["Human_injured"])

        # Train/Test split
        np.random.seed(seed)
        indices = np.random.permutation(len(df))
        n_train = int(len(df) * train_ratio)

        if split == "train":
            sub_idx = indices[:n_train]
        elif split == "test":
            sub_idx = indices[n_train:]
        else:
            sub_idx = indices

        scaler = StandardScaler()
        # Fit on train set
        scaler.fit(df_feat.iloc[indices[:n_train]])
        feat_norm = scaler.transform(df_feat.iloc[sub_idx])

        self.x = torch.tensor(feat_norm, dtype=torch.float32)
        self.y = torch.tensor(df_tgt.iloc[sub_idx].values, dtype=torch.float32)
        self.district_names = df["Dist_Name"].iloc[sub_idx].values
        self.feature_names = feature_cols
        self.target_names = target_cols
        self.scaler = scaler

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def get_impact_loaders(
    csv_path: str = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\processed_data\district_features\district_impact_features.csv",
    batch_size: int = 32
):
    train_ds = DistrictImpactDataset(csv_path, split="train")
    test_ds = DistrictImpactDataset(csv_path, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_ds.scaler

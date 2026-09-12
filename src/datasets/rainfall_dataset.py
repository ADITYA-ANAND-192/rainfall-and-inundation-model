"""
PyTorch Spatiotemporal Dataset for Rainfall Prediction (Model 1)
Loads multi-channel atmospheric reanalysis grids (tp, t2m, d2m, u10, v10)
and pairs them with ground-truth IMD daily rainfall grids.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RainfallGridDataset(Dataset):
    def __init__(
        self,
        npz_path: str = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\processed_data\rainfall_tensors\era5_imd_paired_grid.npz",
        split: str = "train",
        transform=None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        target_transform_log: bool = True
    ):
        super().__init__()
        data = np.load(npz_path, allow_pickle=True)
        features = data["features"]  # [T, 5, 117, 117]
        targets = data["targets"]    # [T, 1, 117, 117]
        self.land_mask = torch.tensor(data["land_mask"], dtype=torch.bool)
        self.latitudes = data["latitudes"]
        self.longitudes = data["longitudes"]
        self.feature_names = list(data["feature_names"])
        self.target_transform_log = target_transform_log

        T = len(features)
        n_train = int(T * train_ratio)
        n_val = int(T * val_ratio)

        # Compute mean & std over train set for channel normalization
        train_feat = features[:n_train]
        self.mean = train_feat.mean(axis=(0, 2, 3), keepdims=True)
        self.std = train_feat.std(axis=(0, 2, 3), keepdims=True) + 1e-6

        # Slice split
        if split == "train":
            feat_split = features[:n_train]
            tgt_split = targets[:n_train]
        elif split == "val":
            feat_split = features[n_train:n_train + n_val]
            tgt_split = targets[n_train:n_train + n_val]
        elif split == "test":
            feat_split = features[n_train + n_val:]
            tgt_split = targets[n_train + n_val:]
        else:
            feat_split = features
            tgt_split = targets

        # Normalize features
        feat_norm = (feat_split - self.mean) / self.std

        self.x = torch.tensor(feat_norm, dtype=torch.float32)
        if self.target_transform_log:
            # log1p transformation for rainfall mm
            self.y = torch.tensor(np.log1p(tgt_split), dtype=torch.float32)
        else:
            self.y = torch.tensor(tgt_split, dtype=torch.float32)

        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

def get_rainfall_loaders(
    npz_path: str = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\processed_data\rainfall_tensors\era5_imd_paired_grid.npz",
    batch_size: int = 16,
    num_workers: int = 0
):
    train_ds = RainfallGridDataset(npz_path=npz_path, split="train")
    val_ds = RainfallGridDataset(npz_path=npz_path, split="val")
    test_ds = RainfallGridDataset(npz_path=npz_path, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds.land_mask

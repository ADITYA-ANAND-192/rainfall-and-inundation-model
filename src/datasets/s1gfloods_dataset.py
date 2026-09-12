"""
PyTorch Dataset for S1GFloods SAR Flood Inundation Segmentation (Model 2)
Pairs pre-flood (A) and post-flood (B) Sentinel-1 SAR tiles with ground-truth
binary flood inundation masks (Label).

Includes:
- 6-channel composite loading [A_RGB, B_RGB] -> [6, 256, 256]
- Siamese dual-tensor loading
- Robust spatial augmentations (horizontal flip, vertical flip, rot90)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms.functional as TF
import random

class S1GFloodsDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        is_train: bool = True,
        augment: bool = True,
        return_siamese: bool = False
    ):
        super().__init__()
        self.df = pd.read_csv(manifest_path)
        self.is_train = is_train
        self.augment = augment and is_train
        self.return_siamese = return_siamese

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path_a = row["path_a"]
        path_b = row["path_b"]
        path_lbl = row["path_label"]

        # Load images
        img_a = Image.open(path_a).convert("RGB")
        img_b = Image.open(path_b).convert("RGB")
        mask = Image.open(path_lbl).convert("L")

        # Convert to Tensor [0.0, 1.0]
        tensor_a = TF.to_tensor(img_a)      # [3, 256, 256]
        tensor_b = TF.to_tensor(img_b)      # [3, 256, 256]
        tensor_lbl = TF.to_tensor(mask)     # [1, 256, 256]

        # Binarize mask
        tensor_lbl = (tensor_lbl > 0.5).float()

        # Augmentations (applied synchronously to A, B, and Mask)
        if self.augment:
            if random.random() > 0.5:
                tensor_a = TF.hflip(tensor_a)
                tensor_b = TF.hflip(tensor_b)
                tensor_lbl = TF.hflip(tensor_lbl)

            if random.random() > 0.5:
                tensor_a = TF.vflip(tensor_a)
                tensor_b = TF.vflip(tensor_b)
                tensor_lbl = TF.vflip(tensor_lbl)

            rot_choice = random.choice([0, 90, 180, 270])
            if rot_choice != 0:
                tensor_a = TF.rotate(tensor_a, rot_choice)
                tensor_b = TF.rotate(tensor_b, rot_choice)
                tensor_lbl = TF.rotate(tensor_lbl, rot_choice)

        if self.return_siamese:
            return tensor_a, tensor_b, tensor_lbl
        else:
            # Concatenate pre- and post-flood images into 6 channels
            tensor_6ch = torch.cat([tensor_a, tensor_b], dim=0) # [6, 256, 256]
            return tensor_6ch, tensor_lbl

def get_s1gfloods_loaders(
    splits_dir: str = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\processed_data\s1gfloods_splits",
    batch_size: int = 16,
    num_workers: int = 0
):
    train_csv = os.path.join(splits_dir, "train.csv")
    val_csv = os.path.join(splits_dir, "val.csv")
    test_csv = os.path.join(splits_dir, "test.csv")

    train_ds = S1GFloodsDataset(train_csv, is_train=True, augment=True)
    val_ds = S1GFloodsDataset(val_csv, is_train=False, augment=False)
    test_ds = S1GFloodsDataset(test_csv, is_train=False, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

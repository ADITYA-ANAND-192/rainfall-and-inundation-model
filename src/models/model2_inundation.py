"""
Model 2: Flood Inundation Prediction Pipeline
Components:
1. Dual-Stream SAR U-Net (Deep Learning Segmentation for S1GFloods)
   Inputs: Pre-flood (A) + Post-flood (B) Sentinel-1 SAR tiles [B, 6, 256, 256]
   Outputs: Binary flood inundation mask [B, 1, 256, 256]
   Loss: Hybrid Dice + BCE Loss
2. Regional Inundation Regressor (predicts district flooded area % from hydro-topography)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class S1FloodUNet(nn.Module):
    """
    U-Net segmentation network for multi-temporal Sentinel-1 SAR flood inundation mapping.
    Accepts 6-channel stacked input (3 channels pre-flood + 3 channels post-flood).
    """
    def __init__(self, in_channels: int = 6, num_classes: int = 1, base_features: int = 32):
        super().__init__()
        # Encoder
        self.inc = DoubleConv(in_channels, base_features)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features, base_features * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features * 2, base_features * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features * 4, base_features * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features * 8, base_features * 16))

        # Decoder with skip connections
        self.up1 = nn.ConvTranspose2d(base_features * 16, base_features * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_features * 16, base_features * 8)

        self.up2 = nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_features * 8, base_features * 4)

        self.up3 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_features * 4, base_features * 2)

        self.up4 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(base_features * 2, base_features)

        # Output head (single-channel logits)
        self.outc = nn.Conv2d(base_features, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up4(x)

        logits = self.outc(x)
        return logits

class DiceBCELoss(nn.Module):
    """
    Combined Binary Cross Entropy + Soft Dice Loss for imbalanced flood segmentation.
    """
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        # BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        
        # Soft Dice
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice
        
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice_loss

def compute_segmentation_metrics(logits, targets, threshold: float = 0.5):
    """
    Computes Intersection-over-Union (IoU), Precision, Recall, and F1 (Dice).
    """
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > threshold).float()
        intersection = (preds * targets).sum().item()
        total_union = ((preds + targets) > 0).float().sum().item()
        
        iou = (intersection + 1e-6) / (total_union + 1e-6)
        precision = (intersection + 1e-6) / (preds.sum().item() + 1e-6)
        recall = (intersection + 1e-6) / (targets.sum().item() + 1e-6)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return {"iou": iou, "precision": precision, "recall": recall, "f1": f1}

"""
S1GFloods Dataset Preparation Pipeline
Indexes the 5,360 Sentinel-1 SAR dual-image pairs (A: pre-flood, B: post-flood)
and ground-truth flood inundation masks (Label: 256x256 binary mask).

Calculates flood fraction per tile and generates stratified Train / Val / Test manifests:
- Train: 70% (3,752 pairs)
- Val: 15% (804 pairs)
- Test: 15% (804 pairs)
"""

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

def prepare_s1gfloods_manifests(
    s1_dir: str = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\05_S1GFloods",
    output_dir: str = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\processed_data\s1gfloods_splits",
    seed: int = 42
):
    os.makedirs(output_dir, exist_ok=True)
    dir_a = os.path.join(s1_dir, "A")
    dir_b = os.path.join(s1_dir, "B")
    dir_lbl = os.path.join(s1_dir, "Label")

    label_files = sorted(glob.glob(os.path.join(dir_lbl, "*.png")))
    print(f"Scanning S1GFloods tiles: found {len(label_files)} label masks.")

    records = []
    print("Computing flood pixel distribution for stratified manifest generation...")
    for idx, lbl_path in enumerate(label_files):
        fname = os.path.basename(lbl_path)
        path_a = os.path.join(dir_a, fname)
        path_b = os.path.join(dir_b, fname)

        if not os.path.exists(path_a) or not os.path.exists(path_b):
            continue

        # Sample read label mask
        with Image.open(lbl_path) as img:
            arr = np.array(img)
            flood_ratio = float((arr > 128).mean())

        # Flood category: 0 = minimal (<1%), 1 = moderate (1-10%), 2 = severe (>10%)
        if flood_ratio < 0.01:
            cat = 0
        elif flood_ratio < 0.10:
            cat = 1
        else:
            cat = 2

        records.append({
            "image_id": fname,
            "path_a": path_a,
            "path_b": path_b,
            "path_label": lbl_path,
            "flood_pct": round(flood_ratio * 100.0, 4),
            "flood_category": cat
        })

    df = pd.DataFrame(records)
    print(f"Successfully cataloged {len(df)} matched triplets.")
    print("Flood category distribution:")
    print(df['flood_category'].value_counts().rename({0: "Minimal (<1%)", 1: "Moderate (1-10%)", 2: "Severe (>10%)"}))

    # Stratified Split: 70% Train, 30% Temp (15% Val, 15% Test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df['flood_category']
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df['flood_category']
    )

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved Train manifest ({len(train_df)} tiles) to: {train_path}")
    print(f"Saved Val manifest   ({len(val_df)} tiles) to: {val_path}")
    print(f"Saved Test manifest  ({len(test_df)} tiles) to: {test_path}")

    # Summary manifest
    summary_path = os.path.join(output_dir, "dataset_summary.json")
    summary = {
        "total_pairs": len(df),
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "mean_flood_pct": float(df['flood_pct'].mean()),
        "max_flood_pct": float(df['flood_pct'].max())
    }
    import json
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return train_path, val_path, test_path

if __name__ == "__main__":
    prepare_s1gfloods_manifests()

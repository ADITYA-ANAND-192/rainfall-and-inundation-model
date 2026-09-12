"""
District Flood Impact & Vulnerability Feature Engineering Pipeline
Combines District Flooded Area, District Flood Impact, demographic data,
and hydro-meteorological indicators into a unified dataset for Model 3
(Infrastructure Impact & Disaster Risk Assessment).

Outputs:
- processed_data/district_features/district_impact_features.csv
- Train / Test splits for tabular impact prediction models
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_impact_features(
    inventory_dir: str = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\02_Flood_Inventory",
    output_dir: str = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\processed_data\district_features",
    seed: int = 42
):
    os.makedirs(output_dir, exist_ok=True)
    area_csv = os.path.join(inventory_dir, "District_FloodedArea.csv")
    impact_csv = os.path.join(inventory_dir, "District_FloodImpact.csv")
    events_csv = os.path.join(inventory_dir, "India_Flood_Inventory_v3.csv")

    print(f"Loading district flooded area: {area_csv}")
    df_area = pd.read_csv(area_csv)

    print(f"Loading district flood impact: {impact_csv}")
    df_impact = pd.read_csv(impact_csv)

    # Merge on Dist_Name
    merged = pd.merge(df_area, df_impact, on="Dist_Name", how="inner")
    print(f"Merged raw district dataset has {len(merged)} records.")

    # Clean duplicates if any by grouping on Dist_Name
    merged = merged.groupby("Dist_Name", as_index=False).agg({
        "Percent_Flooded_Area": "mean",
        "Parmanent_Water": "mean",
        "Corrected_Percent_Flooded_Area": "mean",
        "Human_fatality": "max",
        "Human_injured": "max",
        "Population": "first",
        "Mean_Flood_Duration": "mean"
    })

    # Fill missing duration with median
    duration_median = merged["Mean_Flood_Duration"].median()
    merged["Mean_Flood_Duration"] = merged["Mean_Flood_Duration"].fillna(duration_median)

    # Engineered Features
    # 1. Total casualties
    merged["Total_Casualties"] = merged["Human_fatality"] + merged["Human_injured"]

    # 2. Flooded population proxy
    merged["Flooded_Population_Proxy"] = (merged["Population"] * (merged["Percent_Flooded_Area"] / 100.0)).clip(lower=0)

    # 3. Hazard-Exposure Interaction
    merged["Hazard_Exposure_Product"] = merged["Percent_Flooded_Area"] * np.log1p(merged["Population"])

    # 4. Duration-Weighted Flooding
    merged["Duration_Weighted_Inundation"] = merged["Percent_Flooded_Area"] * merged["Mean_Flood_Duration"]

    # 5. Infrastructure Risk Index (0 - 100)
    # Based on min-max normalized weighted components
    def min_max(s):
        denom = s.max() - s.min()
        return (s - s.min()) / (denom if denom != 0 else 1.0)

    norm_flood = min_max(merged["Percent_Flooded_Area"])
    norm_pop = min_max(np.log1p(merged["Population"]))
    norm_dur = min_max(merged["Mean_Flood_Duration"])
    norm_perm_water = min_max(merged["Parmanent_Water"])

    merged["Infrastructure_Risk_Score"] = (
        norm_flood * 40.0 +
        norm_pop * 30.0 +
        norm_dur * 20.0 +
        norm_perm_water * 10.0
    ).round(2)

    # Categorical Risk Level
    merged["Risk_Level"] = pd.qcut(
        merged["Infrastructure_Risk_Score"],
        q=4,
        labels=["Low", "Moderate", "High", "Critical"]
    )

    out_csv = os.path.join(output_dir, "district_impact_features.csv")
    merged.to_csv(out_csv, index=False)
    print(f"Saved {len(merged)} district impact profiles to: {out_csv}")

    # Train / Test split (80/20)
    train_df, test_df = train_test_split(merged, test_size=0.20, random_state=seed, stratify=merged["Risk_Level"])
    train_path = os.path.join(output_dir, "train_impact.csv")
    test_path = os.path.join(output_dir, "test_impact.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved Train set ({len(train_df)} districts) to {train_path}")
    print(f"Saved Test set  ({len(test_df)} districts) to {test_path}")

    return out_csv, train_path, test_path

if __name__ == "__main__":
    prepare_impact_features()

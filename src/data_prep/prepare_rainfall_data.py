"""
Rainfall Dataset Preparation Pipeline
Aligns ERA5 atmospheric variables (t2m, d2m, u10, v10, tp) with
IMD ground-truth gridded daily rainfall on the exact 0.25x0.25 degree
Indian spatial grid (117 x 117).

Outputs:
1. Spatiotemporal Tensor cache (.npz) containing normalized multi-channel
   features, target rainfall fields, and land-sea binary masks.
2. Tabular feature dataset (.parquet/.csv) for fast Gradient Boosting & Random Forest baselines.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr

def prepare_rainfall_dataset(
    era5_path: str = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\processed_data\era5_extracted\era5_india_daily_2017_2018.nc",
    imd_dir: str = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\01_IMD_Rainfall",
    output_dir: str = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\processed_data\rainfall_tensors"
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading consolidated ERA5 dataset from: {era5_path}")
    ds_era = xr.open_dataset(era5_path)

    # Load IMD 2017 and 2018
    imd_files = [os.path.join(imd_dir, "2017.nc"), os.path.join(imd_dir, "2018.nc")]
    print(f"Loading IMD ground-truth rainfall from: {imd_files}")
    imd_ds_list = []
    for f in imd_files:
        if os.path.exists(f):
            ds = xr.open_dataset(f)
            # Standardize IMD coordinate names to lowercase
            ds = ds.rename({"TIME": "time", "LATITUDE": "latitude", "LONGITUDE": "longitude"})
            # Slice to ERA5 bounding box
            sub = ds.sel(latitude=slice(8.0, 37.0), longitude=slice(68.0, 97.0))
            imd_ds_list.append(sub)
    
    ds_imd = xr.concat(imd_ds_list, dim="time")

    # Find common dates
    era_dates = pd.to_datetime(ds_era.time.values).normalize()
    imd_dates = pd.to_datetime(ds_imd.time.values).normalize()
    common_dates = era_dates.intersection(imd_dates)
    print(f"Found {len(common_dates)} intersecting daily timestamps between ERA5 and IMD.")

    # Select common dates
    ds_era_common = ds_era.sel(time=common_dates)
    ds_imd_common = ds_imd.sel(time=common_dates)

    # Coordinates
    lats = ds_era_common.latitude.values
    lons = ds_era_common.longitude.values
    H, W = len(lats), len(lons)
    T = len(common_dates)
    print(f"Spatial Grid: {H} latitudes x {W} longitudes across {T} days.")

    # Variables: tp (m -> convert to mm by * 1000), t2m (K), d2m (K), u10 (m/s), v10 (m/s)
    tp = (ds_era_common['tp'].values * 1000.0).astype(np.float32)  # mm
    t2m = ds_era_common['t2m'].values.astype(np.float32)
    d2m = ds_era_common['d2m'].values.astype(np.float32)
    u10 = ds_era_common['u10'].values.astype(np.float32)
    v10 = ds_era_common['v10'].values.astype(np.float32)

    # Stack into [T, C=5, H, W]
    features = np.stack([tp, t2m, d2m, u10, v10], axis=1)

    # Target IMD Rainfall [T, 1, H, W]
    target_imd = ds_imd_common['RAINFALL'].values[:, np.newaxis, :, :].astype(np.float32)

    # Compute land mask (where IMD has valid data on land)
    valid_counts = (~np.isnan(target_imd)).sum(axis=(0, 1))
    land_mask = (valid_counts > 0)
    print(f"Land mask contains {land_mask.sum()} active land grid cells out of {H * W} total grid points.")

    # Fill NaN over target with 0.0 for stable loss masking
    target_clean = np.where(np.isnan(target_imd), 0.0, target_imd)
    # Ensure non-negative
    target_clean = np.maximum(target_clean, 0.0)

    # Save NPZ bundle
    npz_path = os.path.join(output_dir, "era5_imd_paired_grid.npz")
    print(f"Saving spatiotemporal tensor archive to: {npz_path}")
    np.savez_compressed(
        npz_path,
        features=features,
        targets=target_clean,
        land_mask=land_mask,
        dates=common_dates.strftime("%Y-%m-%d").values,
        feature_names=["tp_era5_mm", "t2m_k", "d2m_k", "u10_ms", "v10_ms"],
        latitudes=lats,
        longitudes=lons
    )
    print(f"Saved {os.path.getsize(npz_path) / (1024 * 1024):.2f} MB to {npz_path}")

    # Build Tabular sample matrix for ML baselines
    print("Extracting tabular feature matrix for machine learning baselines...")
    sample_dfs = []
    # Sample every 5th land cell across time to keep tabular size optimal
    sampled_indices = np.argwhere(land_mask)
    sampled_indices = sampled_indices[::4] # Sample 25% of land points for tabular training
    
    rows = []
    dates_str = common_dates.strftime("%Y-%m-%d").values
    doy = common_dates.dayofyear.values
    doy_sin = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    doy_cos = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)

    for t_idx, d_str in enumerate(dates_str):
        for lat_idx, lon_idx in sampled_indices:
            imd_val = target_clean[t_idx, 0, lat_idx, lon_idx]
            tp_val = tp[t_idx, lat_idx, lon_idx]
            t2m_val = t2m[t_idx, lat_idx, lon_idx]
            d2m_val = d2m[t_idx, lat_idx, lon_idx]
            u10_val = u10[t_idx, lat_idx, lon_idx]
            v10_val = v10[t_idx, lat_idx, lon_idx]
            wspd = np.sqrt(u10_val**2 + v10_val**2)
            dew_dep = t2m_val - d2m_val
            
            rows.append({
                "date": d_str,
                "lat": lats[lat_idx],
                "lon": lons[lon_idx],
                "doy_sin": doy_sin[t_idx],
                "doy_cos": doy_cos[t_idx],
                "tp_era5": tp_val,
                "t2m": t2m_val,
                "d2m": d2m_val,
                "u10": u10_val,
                "v10": v10_val,
                "wind_speed": wspd,
                "dew_depression": dew_dep,
                "rainfall_imd": imd_val
            })

    df_tab = pd.DataFrame(rows)
    tab_path = os.path.join(output_dir, "rainfall_tabular_features.csv")
    df_tab.to_csv(tab_path, index=False)
    print(f"Saved {len(df_tab)} tabular rows to: {tab_path}")

    # Generate Train/Val/Test split metadata
    split_meta = {
        "total_days": T,
        "train_days": int(T * 0.70),
        "val_days": int(T * 0.15),
        "test_days": T - int(T * 0.70) - int(T * 0.15)
    }
    print(f"Split metadata: {split_meta}")
    ds_era.close()
    ds_imd.close()
    return npz_path, tab_path

if __name__ == "__main__":
    prepare_rainfall_dataset()

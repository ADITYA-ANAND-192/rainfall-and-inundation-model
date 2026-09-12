"""
ERA5 Reanalysis Extraction & Consolidation Pipeline
Unpacks the 15 monthly zip-packaged NetCDF files in 06_ERA5_Dataset,
extracts the 5 meteorological variables (tp, t2m, d2m, u10, v10),
standardizes time and coordinate dimensions, and merges them into
a single, unified NetCDF dataset covering 2017-01 to 2018-03.
"""

import os
import glob
import io
import zipfile
import xarray as xr
import numpy as np

def extract_and_merge_era5(
    input_dir: str = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\06_ERA5_Dataset",
    output_path: str = r"c:\Users\HP\OneDrive\Desktop\Flood_AI_Dataset\processed_data\era5_extracted\era5_india_daily_2017_2018.nc"
):
    print(f"Scanning ERA5 archives in: {input_dir}")
    zip_files = sorted(glob.glob(os.path.join(input_dir, "*.nc")))
    print(f"Found {len(zip_files)} monthly ERA5 archive files.")

    if not zip_files:
        raise FileNotFoundError(f"No ERA5 archives found in {input_dir}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    var_map = {
        "total_precipitation": "tp",
        "2m_temperature": "t2m",
        "2m_dewpoint_temperature": "d2m",
        "10m_u_component_of_wind": "u10",
        "10m_v_component_of_wind": "v10"
    }

    monthly_datasets = []

    for zf_path in zip_files:
        fname = os.path.basename(zf_path)
        print(f"Processing {fname}...")
        with zipfile.ZipFile(zf_path, 'r') as z:
            month_sub_ds = []
            for item in z.infolist():
                if not item.filename.endswith(".nc"):
                    continue
                # Identify variable
                matched_var = None
                for key in var_map:
                    if key in item.filename:
                        matched_var = var_map[key]
                        break
                
                with z.open(item) as f:
                    content = io.BytesIO(f.read())
                    sub_ds = xr.open_dataset(content, engine="h5netcdf").load()
                    
                    # Standardize time coordinate
                    if "valid_time" in sub_ds.coords:
                        sub_ds = sub_ds.rename({"valid_time": "time"})
                    elif "time" not in sub_ds.coords:
                        time_coord = [c for c in sub_ds.coords if "time" in c.lower()]
                        if time_coord:
                            sub_ds = sub_ds.rename({time_coord[0]: "time"})
                    
                    # Remove extraneous scalar coordinates like 'number' if present
                    if "number" in sub_ds.coords:
                        sub_ds = sub_ds.drop_vars("number")

                    month_sub_ds.append(sub_ds)

            if month_sub_ds:
                merged_month = xr.merge(month_sub_ds)
                monthly_datasets.append(merged_month)

    print(f"Concatenating {len(monthly_datasets)} monthly datasets across time...")
    full_ds = xr.concat(monthly_datasets, dim="time")
    full_ds = full_ds.sortby("time")

    # Sort coordinates for consistent spatial indexing
    if "latitude" in full_ds.coords and full_ds.latitude.values[0] > full_ds.latitude.values[-1]:
        full_ds = full_ds.reindex(latitude=full_ds.latitude[::-1])
    full_ds = full_ds.sortby("longitude")

    print("\nConsolidated ERA5 Dataset:")
    print(full_ds)
    print(f"\nSaving consolidated NetCDF to: {output_path}")
    full_ds.to_netcdf(output_path, engine="netcdf4")
    print(f"Successfully saved {os.path.getsize(output_path) / (1024 * 1024):.2f} MB to {output_path}")
    return output_path

if __name__ == "__main__":
    extract_and_merge_era5()

# 🌊 3-Tier Geospatial Flood AI System

An end-to-end multi-tier deep learning and machine learning system for:
1. **Tier 1 (Model 1)**: Spatiotemporal Rainfall Prediction & Downscaling (Atmosphere -> Daily Precipitation Grid)
2. **Tier 2 (Model 2)**: Satellite Remote Sensing Flood Inundation Segmentation (Sentinel-1 SAR -> Inundation Mask)
3. **Tier 3 (Model 3)**: Infrastructure Impact & Disaster Risk Assessment (Rainfall + Inundation -> Fatalities, Injuries & Risk Index)

---

## 🏛️ System Architecture

```text
Atmospheric Reanalysis (ERA5: tp, t2m, d2m, u10, v10)
                 │
                 ▼
    [Model 1: SpatiotemporalRainfallNet]
                 │
                 ▼ Predicted Rainfall Map (mm/day)
    [Model 2: S1FloodUNet (SAR Dual-Stream)]  <── Sentinel-1 SAR (Pre/Post)
                 │
                 ▼ Predicted Inundation Mask & Flooded Area %
    [Model 3: ImpactMultiTaskNet]              <── District Exposure (Population, Duration)
                 │
                 ▼
    Predicted Fatalities, Injuries & Infrastructure Risk Score (0-100)
```

---

## 📊 Dataset Overview

| Component | Source | Coverage / Resolution | Description |
| :--- | :--- | :--- | :--- |
| **ERA5 Reanalysis** | ECMWF / Copernicus CDS | India [8°N–37°N, 68°E–97°E], 0.25° | 5 variables: `tp`, `t2m`, `d2m`, `u10`, `v10` across 455 days (2017–2018). |
| **IMD Rainfall** | Indian Meteorological Dept | All India, 0.25° daily grid | Ground truth gridded daily rainfall in mm (4,958 land cells). |
| **S1GFloods** | Sentinel-1 C-band SAR | 5,360 matched 256×256 tiles | Dual-temporal SAR imagery (`A` pre-flood, `B` post-flood) with binary flood water ground truth (`Label`). |
| **Flood Inventory** | NDMA / Government of India | 726 Districts across India | Flooded area %, permanent water, fatalities, injuries, population, and flood duration. |
| **DEM SRTM** | NASA / USGS SRTM | 3-arcsecond (~90m) resolution | Digital elevation model covering North, Central, and South regions of India. |

---

## 🧠 Model Architectures & Performance

### Tier 1: Rainfall Prediction (`src/models/model1_rainfall.py`)
- **Architecture**: `SpatiotemporalRainfallNet` with dilated residual convolutions preserving geographic coordinates.
- **Loss**: `MaskedLoss` (Huber loss computed strictly over Indian land territory).
- **Test Performance (69 unseen days)**:
  - **RMSE**: 2.683 mm/day
  - **MAE**: 0.449 mm/day
  - **Pearson $r$**: 0.416
  - Companion Tabular GBDT: **RMSE = 2.359 mm, MAE = 0.573 mm**

### Tier 2: Flood Inundation Segmentation (`src/models/model2_inundation.py`)
- **Architecture**: `S1FloodUNet` (4-level dual-stream encoder-decoder U-Net with skip connections).
- **Loss**: `DiceBCELoss` (hybrid binary cross-entropy + soft Dice loss).
- **Test Performance (Unseen SAR test tiles)**:
  - **Mean IoU**: **74.9% (0.7488)**
  - **Mean Dice Score / F1**: **84.1% (0.8412)**
  - **Precision**: 80.8%
  - **Recall**: 91.0%

### Tier 3: Infrastructure Impact Assessment (`src/models/model3_impact.py`)
- **Architecture**: `ImpactMultiTaskNet` (multi-task network with shared layers and distinct casualty/risk heads).
- **Test Performance (146 unseen districts)**:
  - **Infrastructure Risk Score**: **$R^2 = 0.891$**, **RMSE = 3.26 points**, **MAE = 2.65 points** (0–100 scale)
  - **Human Injured**: MAE = 9.04 injured
  - **Human Fatalities**: MAE = 57.85 casualties

---

## 🚀 Getting Started

### 1. Installation
```bash
git clone <repo-url>
cd Flood_AI_Dataset
pip install torch torchvision numpy pandas xarray netCDF4 scikit-learn pillow imagecodecs
```

### 2. Preprocess Datasets
```bash
# 1. Unpack & merge ERA5
python src/data_prep/extract_era5.py

# 2. Align ERA5 with IMD Rainfall
python src/data_prep/prepare_rainfall_data.py

# 3. Create stratified S1GFloods manifests (70/15/15)
python src/data_prep/prepare_s1gfloods.py

# 4. Feature engineering for district disaster impacts
python src/data_prep/prepare_impact_data.py
```

### 3. Run Automated Integration Tests
```bash
python tests/test_pipelines.py
```

### 4. Train Models
```bash
# Train Model 1 (Rainfall)
python train_all.py --model 1 --epochs 5 --batch_size 16

# Train Model 2 (Flood Inundation U-Net)
python train_all.py --model 2 --epochs 5 --batch_size 8

# Train Model 3 (Impact Assessment)
python train_all.py --model 3 --epochs 10 --batch_size 32

# Train all 3 sequentially
python train_all.py --model all --epochs 3 --batch_size 8
```

### 5. Run Chained End-to-End Simulation
```bash
python src/pipeline/end_to_end_infer.py
```

### 6. Run Complete Evaluation Suite
```bash
python evaluate_all_models.py
```

---

## 📁 Repository Structure
```text
Flood_AI_Dataset/
├── processed_data/               # Preprocessed manifests and tabular features
│   ├── s1gfloods_splits/         # train.csv, val.csv, test.csv
│   └── district_features/        # district_impact_features.csv, train_impact.csv, test_impact.csv
├── src/
│   ├── data_prep/                # Extraction and preprocessing pipelines
│   ├── datasets/                 # PyTorch Dataset loaders
│   ├── models/                   # Neural network architectures
│   └── pipeline/                 # Chained inference pipeline
├── models_checkpoints/           # Trained PyTorch weights (.pt)
├── tests/                        # Automated unit and integration tests
├── evaluate_all_models.py        # Complete evaluation script
├── train_all.py                  # Master training CLI
└── README.md
```

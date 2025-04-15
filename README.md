
# 🌇 Sub-Meter Urban Surface Albedo Prediction for 34 U.S. Cities based on Deep Learning

This repository contains the full codebase, data pipeline, and model implementation for the project:

**"Sub-Meter Surfaces Albedo Mapping across 34 Major U.S. Cities Using Deep Learning and Multisource Remote Sensing Data"**

## 🔍 Overview

Urban surface albedo plays a critical role in shaping local microclimates. However, existing albedo datasets (e.g., MODIS, Sentinel-2) offer only 10–30m resolution, limiting their applicability for micro-scale thermal modeling and heat exposure assessments.

This project develops a deep learning framework that predicts **1-meter resolution albedo** for **impervious** and **pervious** surfaces using multisource geospatial data and semantic segmentation models. Outputs can support neighborhood-scale environmental planning, heat mitigation, and climate adaptation strategies.

---

## 📁 Project Structure

```
Sub-meter-albedo-prediction/
│
├── 1_data_preprocessing/        # Prepare NAIP tiles, building masks, albedo clipping
├── 2_training_dataset/          # Generate training image-label pairs (ISA, PSA)
├── 3_model_training/            # Train U-Net models for ISA & PSA
├── 4_prediction_inference/      # Patch-wise prediction & mosaicking across 34 cities
├── 5_evaluation/                # Evaluation metrics (R², MAE, RMSE) & visualization
└── README.md                    # Project introduction (this file)
```

---

## 🌐 Data Sources

| Dataset                    | Source                                           | Resolution | Purpose                                   |
|---------------------------|--------------------------------------------------|------------|-------------------------------------------|
| **NAIP imagery**          | USDA FSA via EarthExplorer                      | 0.6 m      | Main input for model                      |
| **Roof albedo polygons**  | Berkeley Lab Heat Island Group                  | Polygon    | Ground truth for ISA                      |
| **Sentinel-2 (L2A)**      | Google Earth Engine                             | 10 m       | Labels for pervious surface albedo (PSA)  |
| **Building footprints**   | Microsoft / LARIAC4                              | Polygon    | Mask generation for impervious areas      |
| **Land cover (MULC/ESA)** | University of Vermont SAL / ESA WorldCover      | 1 m / 10 m | ISA/PSA label filtering                   |

---

## 🧠 Model Summary

- **Architecture**: U-Net
- **Tasks**:
  - ISA (Impervious Surface Albedo)
  - PSA (Pervious Surface Albedo)
- **Input**: 512×512 NAIP patches
- **Output**: Pixel-wise albedo predictions
- **Training**:
  - Optimizer: Adam
  - Epochs: 100
  - Batch size: 4
  - Loss: MSE
- **Post-processing**:
  - Mosaicking 1024×1024 patches
  - Buffer removal
  - Overlay building polygons

---

## 🧪 Evaluation Metrics

- Coefficient of Determination (R²)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Visual comparison of predicted vs. ground truth tiles
- Cross-city generalizability tests

---

## 🛠️ Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.10
- NumPy, Pandas, Rasterio, OpenCV
- CUDA-enabled GPU for training

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 1. Preprocess Raw Data
```bash
python 1_data_preprocessing/prepare_naip_tiles.py
```

### 2. Generate Training Samples
```bash
python 2_training_dataset/create_training_patches.py
```

### 3. Train ISA or PSA Model
```bash
python 3_model_training/train_isa.py
# or
python 3_model_training/train_psa.py
```

### 4. Predict Full City-Scale Albedo
```bash
python 4_prediction_inference/predict_city_albedo.py
```

### 5. Evaluate and Visualize
```bash
python 5_evaluation/metrics_summary.py
```

---

## 📊 Results Summary

| Task | R²   | MAE  | RMSE |
|------|------|------|------|
| ISA  | 0.865 | 0.042 | 0.058 |
| PSA  | 0.814 | 0.051 | 0.067 |

---

## 📘 Citation

If you use this work, please cite:

*Coming soon*

---

## 🙌 Acknowledgements

This work was supported by high-resolution data from USDA, Sentinel-2 via Google Earth Engine, Berkeley Lab, and Microsoft. We thank our collaborators for their feedback and contributions.

---

## 📬 Contact

For questions or suggestions, contact: **Shengao Yi** at [shengao@upenn.edu]

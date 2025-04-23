
# 🌇 Sub-Meter Urban Surface Albedo Prediction for 34 U.S. Cities based on Deep Learning

This repository contains the full pipeline and source code for our paper:

**"A Sub-Meter Resolution Urban Surfaces Albedo Dataset for 34 U.S. Cities based on Deep Learning Network"**

## 🌟 Project Summary

Accurately mapping urban surface albedo at a fine spatial scale is crucial for microclimate modeling and urban heat mitigation planning. This project presents the first high-resolution (0.6m) albedo dataset for **impervious** and **pervious** surfaces across 34 major U.S. cities.

We utilized NAIP imagery, roof albedo ground truth, Sentinel-2 data, and deep learning models (U-Net variants) to predict **impervious surface albedo (ISA)** and **pervious surface albedo (PSA)**.

📍**Project Website & Visualization**:  
👉 https://shengaoyi.github.io/#/Albedopedia

📄 **Reviewed by**: *Scientific Data* (Nature Portfolio)

👨‍💻 **Authors**: Shengao Yi*, Xiaojiang Li, Yixuan Liu, Xinyu Dong, Wei Tu

---

## 🔍 Overview

Albedo refers to the fraction of incoming solar radiation that is reflected by a surface, ranging from 0 (no reflection) to 1 (total reflection). It is a key parameter in understanding how different land surfaces interact with solar energy. Surface albedo is an essential parameter which quantifies the amount of solar radiation reflected by the Earth’s surface. It plays a critical role in shaping local microclimates. However, existing albedo datasets (e.g., MODIS, Sentinel-2) offer only 10–30m resolution, limiting their applicability for micro-scale thermal modeling and heat exposure assessments.

This project develops a deep learning framework that predicts **1-meter resolution albedo** for **impervious** and **pervious** surfaces using multisource geospatial data and semantic segmentation models. Outputs can support neighborhood-scale environmental planning, heat mitigation, and climate adaptation strategies.

---

## 📁 Project Structure

```
Sub-meter-albedo-prediction/
├── 1_data_preprocessing/        # Prepare NAIP tiles, building masks, albedo clipping
├── 2_training_dataset/          # Generate training image-label pairs (ISA, PSA)
├── 3_model_training/            # Train U-Net models for ISA & PSA
├── 4_prediction_inference/      # Patch-wise prediction & mosaicking across 34 cities
├── 5_evaluation/                # Evaluation metrics (R², MAE, RMSE) & visualization
└── README.md                    # Project introduction (this file)
```

---

## Study Area

The study covers 34 major U.S. cities, enabling a comprehensive analysis of surface albedo at sub-meter resolution. This geographic diversity captures a wide range of urban morphologies, land cover patterns, and climatic zones.

![Image](https://github.com/user-attachments/assets/a5b05bb9-4eaa-40b9-8d64-91c0fe9ccd34)
*Figure 1. Study area: 34 selected U.S. cities.*

---

## Overall Framework

![Image](https://github.com/user-attachments/assets/f8e100a9-f32e-4b78-864c-349f04a26050)
*Figure 2. Overall framework for surface albedo mapping: 1) collecting diverse data sources; 2) preprocessing for model readiness; 3) model training and evaluating for accurate urban albedo mapping.*

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

## NAIP Maps
![Image](https://github.com/user-attachments/assets/d8aa0cda-900c-45cb-8f33-e45dd1c091f6)
*Figure 3. NAIP maps of 34 major U.S. cities.*

---
## Roof Albedo
![Figure 4 Roof_Albedo](https://github.com/user-attachments/assets/8ff58950-ad4e-45f2-a467-6a4b30435791)
*Figure 4. The spatial distribution of impervious roof albedo in 4 U.S. cities.*

---
## UNet
![Image](https://github.com/user-attachments/assets/dc868f6a-4d56-4555-b345-b865a1be9252)
*Figure 5. U-Net framework for impervious surface classification and albedo prediction.*

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
| ISA  | 0.9028 | 0.0057 | 0.0002 |
| PSA  | 0.9538 | 0.0027 | 0.00002 |

![Image](https://github.com/user-attachments/assets/4e0cc4a7-48f6-4573-8b49-62cafebcce68)
*Figure 6. Urban surface albedo maps of 34 major U.S. cities.*

![Image](https://github.com/user-attachments/assets/324fe2ed-f9d1-48c1-84e5-69e2e1effe49)
*Figure 7. Illustrative example of urban albedo mapping process and its precise results.*

---

## 📘 Citation

If you use this work, please cite:

*Yi, S., Li, X., Liu, Y., Dong, X. & Tu, W. A sub-meter resolution urban surface albedo dataset for 34 U.S. cities based on deep learning. Dataset, https://doi.org/10.6084/m9.figshare.27850965 (2024).*

---

## 🙌 Acknowledgements

This work was supported by high-resolution data from USDA, Sentinel-2 via Google Earth Engine, Berkeley Lab, and Microsoft. We thank our collaborators for their feedback and contributions.

---

## 📬 Contact

For questions or suggestions, contact: **Shengao Yi** at [shengao@upenn.edu]


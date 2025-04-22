
# 🌇 Sub-Meter Urban Surface Albedo Prediction for 34 U.S. Cities based on Deep Learning

This repository contains the full pipeline and source code for our paper:

**"A Sub-Meter Resolution Urban Surfaces Albedo Dataset for 34 U.S. Cities based on Deep Learning Network"**

📍**Project Website & Visualization**:  
👉 https://shengaoyi.github.io/#/Albedopedia

📄 **Reviewed by**: *Scientific Data* (Nature Portfolio)

👨‍💻 **Authors**: Shengao Yi*, Xiaojiang Li, Yixuan Liu, Xinyu Dong, Wei Tu

---

## 🌟 Project Summary

Accurately mapping urban surface albedo at a fine spatial scale is crucial for microclimate modeling and urban heat mitigation planning. This project presents the first high-resolution (0.6m) albedo dataset for **impervious** and **pervious** surfaces across 34 major U.S. cities.

We utilized NAIP imagery, roof albedo ground truth, Sentinel-2 data, and deep learning models (U-Net variants) to predict **impervious surface albedo (ISA)** and **pervious surface albedo (PSA)**.

---

## 🔍 Overview

Urban surface albedo plays a critical role in shaping local microclimates. However, existing albedo datasets (e.g., MODIS, Sentinel-2) offer only 10–30m resolution, limiting their applicability for micro-scale thermal modeling and heat exposure assessments.

This project develops a deep learning framework that predicts **1-meter resolution albedo** for **impervious** and **pervious** surfaces using multisource geospatial data and semantic segmentation models. Outputs can support neighborhood-scale environmental planning, heat mitigation, and climate adaptation strategies.

### 🖼️ Key Figures
- **Figure 1. Framework Overview**  
  ![Framework](./figures/Figure%201%20Framework.jpg)
- **Figure 2. Study Cities in the U.S.**  
  ![US Cities](./figures/Figure%202%20US_Cities.jpg)
- **Figure 3. NAIP Image Sample**  
  ![NAIP](./figures/Figure%203%20NAIP.jpg)
- **Figure 4. Roof Albedo Ground Truth**  
  ![Roof Albedo](./figures/Figure%204%20Roof_Albedo.jpg)
- **Figure 5. U-Net Model Architecture**  
  ![UNet](./figures/Figure%205%20U-Net.jpg)
- **Figure 6. Predicted Albedo for 34 Cities**  
  ![Cities Albedo](./figures/Figure%206%20Surface_Albedo_Cities.jpg)
- **Figure 7. ISA Prediction Example**  
  ![ISA](./figures/Figure%207%20ISA.jpg)
- **Figure 8. PSA Prediction Example**  
  ![PSA](./figures/Figure%208%20PSA.jpg)
- **Figure 9. Philadelphia Albedo Map**  
  ![Philadelphia](./figures/Figure%209%20Philadelphia_Albedo.jpg)
- **Figure 10. Albedo Distribution Statistics**  
  ![Stats](./figures/Figure%2010%20Albedo_Statistics.jpg)

(Ensure your local `figures/` folder contains these files for correct rendering.)

---

## 📁 Project Structure

```
Sub-meter-albedo-prediction/
├── 1_data_preprocessing/
├── 2_training_dataset/
├── 3_model_training/
├── 4_prediction_inference/
├── 5_evaluation/
└── README.md
```

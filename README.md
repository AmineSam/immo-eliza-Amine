# Immo-Eliza — Real-Estate Price Predictor (Belgium)

## 1. Project Overview

Immo-Eliza is a **fully modular, leakage-safe machine-learning system** designed to predict real-estate prices in Belgium using enriched geographic, socio‑economic, OSM distance, clustering, and advanced ML features.

The project includes:

- Multi-stage preprocessing pipeline  
- Geographic + economic enrichment  
- Specialized models for **houses** and **apartments** 
- Reduced‑feature modelling for user friendly web app later  
- GPU‑accelerated Optuna optimization (XGBoost + CatBoost) on Kaggle  
- Ensemble tuning and robust cross-validation  

---

## 2. Repository Structure

```
immo-eliza-Amine/
│
├── config/
│   ├── paths.py
│   └── settings.py
│   └── constants.py
│
├── data/
│   ├── raw/
│   ├── stage1/
│   ├── stage2/
│   ├── stage3/
│   └── clean/
│
├── models/
│   ├── model_xgb_apartment.pkl
│   ├── model_xgb_house.pkl
│   ├── stage3_pipeline_apartment.pkl
│   ├── stage3_pipeline_house.pkl
│
├── notebooks/
│   └── analysis.ipynb
│   └── specilized_models.ipynb
│
├── pipelines/
│   ├── export_for_kaggle.py
│   ├── stage1_basic_cleaning.py
│   ├── stage2_plausibility_outliers_missing.py
│   ├── stage2_5_geo_enrichment.py
│   ├── stage3_feature_engineering.py
│   ├── stage3_fitted.py
│   └── pipeline_runner.py
│   └── stage3_fitted.py
│
└── utils/
│   └── ml_utils.py
│   └── apartment_optimization_gpu.py
│   └── house_optimization_gpu.py
│   └── optuna_tuning_v5.py
│   └── stage3_utils.py
```

---

## 3. High-Level Pipeline Architecture

```
Stage 1 → Stage 2 → Stage 2.5  → Split → Stage 3 Fit → Stage 3 Transform
```

### Key Principles
- Domain-based thresholds 
- Train/val/test split **before** any fitted transformation  
- Imputers, encoders, TE: fit only on training  

---

## 4. Preprocessing Overview

### Stage 1 — Base Cleaning
- Raw loading  
- Extraction from URL fields  
- Normalization of binary & numeric data  

### Stage 2 — Plausibility & Outlier Filtering
- Domain thresholds (price, area, EPC, surfaces, rooms…)  
- Removes inconsistencies & data errors  

### Stage 2.5 — Geographic & Socio‑Economic Enrichment
- Region, province, arrondissement, municipality  
- Median income  
- Benchmark price-per-m² (national/regional/provincial)  
- Locality & postal aggregates  
  

### Stage 3 — Final Modelling Features
- Missingness indicators  
- Log transforms  
- Target encoding (smoothed)  

---

## 5. Specialized Models

### House-Only Model
 

### Apartment-Only Model

---

## 6. Reduced-Feature Model (User friendly inputs)


---

## 7. Machine Learning Models

Implemented:
- **XGBoost** (primary general model)   R²: 0.86 and MAE: 44510 eur (5 CV)
- **CatBoost** (GPU‑optimized strong alternative)   R²: 0.85 and MAE: 43703 eur (5 CV)
- Random Forest (baseline)  R²: 0.83 and MAE: 49437 eur (5 CV)
- Linear Regression (baseline)   R²: 0.60 and MAE: 90403 eur (5 CV)

### Ensemble
Weighted average of tuned XGB + tuned CAT  
Optimized with:
- 5-fold CV  
- Gap constraint: R² gap < 0.1
- Best results for 60% XGBoost and 40% CatBoost
  - slightly improved R²(0.86)
  - reduced gap of overfitting (less than 0.12 between R² in train and test)

---

## 8. Kaggle GPU Optimization (Optuna)

A dedicated GPU script performs:
- Optuna hyperparameter search  
- Early stopping & pruning  
- MAE-centric objective  
- R²-centric objective
- Constraint on generalization gap  

Models Tuned:
- XGBoost  
- CatBoost  

---

## 9. How to Run
```bash
python -m venv .venv
. .venv/bin/activate  # or Scripts\activate on Windows
pip install -r requirements.txt
```
```python
from pipelines.export_for_kaggle import export_for_kaggle  
from utils.predictor import predict_price
from utils.stage3_utils import (
    fit_stage3,
    transform_stage3,
    prepare_X_y
)
export_for_kaggle()
predict_price(house_test)
predict_price(apartment_test)
```


## 10. Future Work

- Deep-learning price models  
- Transformer-based geospatial embeddings  
- H3 spatial hierarchy encoding  
- Time-aware price dynamics  
- Streamlit or FastAPI deployment  
- Automated monitoring & drift detection  

---

## 11. Author

**Amine Samoudi**  
GitHub: https://github.com/AmineSam

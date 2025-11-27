# Immo-Eliza — Advanced Real-Estate Data Pipeline (Belgium)

## 1. Project Overview

This project implements a **robust, leakage-safe, end-to-end data pipeline** for predicting real-estate prices in Belgium.  
The dataset is enriched with:

- Geographic hierarchy (municipality, arrondissement, province, region)  
- Socio-economics (median municipal income)  
- Address-level fields  
- National, regional, and provincial benchmark prices per m²  
- Local price aggregates (postal, locality)  
- Advanced engineered ML features (missingness flags, log transforms, smoothed target encoding)

The pipeline is designed to be **modular, reproducible, and model-agnostic**, allowing training of models such as Linear Regression, Random Forest, and XGBoost.

---

## 2. Repository Structure

```
immo-eliza-Amine/
│
├── config/
│   ├── paths.py
│   └── settings.py
│
├── data/
│   ├── raw/
│   ├── stage1/
│   ├── stage2/
│   ├── stage3/
│   └── clean/
│
├── notebooks/
│   └── analysis.ipynb
│
├── pipelines/
│   ├── stage0_load_raw.py
│   ├── stage1_basic_cleaning.py
│   ├── stage2_plausibility_outliers_missing.py
│   ├── stage2_5_geo_enrichment.py
│   ├── stage3_feature_engineering.py
│   ├── stage3_fitted.py
│   └── pipeline_runner.py
│
└── utils/
    └── ml_utils.py
```

---

## 3. End-to-End Pipeline Architecture

The processing workflow is structured into **strict leakage-safe stages**:

```
Stage 0 → Stage 1 → Stage 2 → Stage 2.5 → Stratified Split → Stage 3 (Fit/Transform)
```

---

## 4. Stage-by-Stage Documentation

### 4.1 Stage 0 — Load Raw Dataset

- Load CSV  
- Remove duplicates  
- Validate schema  
- Enforce core dtypes  
- Drop invalid rows  

### 4.2 Stage 1 — URL-Based Extraction & Basic Cleaning

- Extract postal code, locality, subtype  
- Clean numeric/Yes–No fields  
- Normalize property subtype  
- Map to Apartment / House / Other  

### 4.3 Stage 2 — Plausibility Checks, Encoding, Outlier Removal

- Drop sparse/noisy columns  
- Encode categorical & binary fields  
- Replace missing numeric with -1  
- Apply plausibility rules  
- Split out outliers  

### 4.4 Stage 2.5 — Geographic & Socio-Economic Enrichment

Adds:

- Municipality, arrondissement, province, region  
- Median municipal income  
- Address table  
- Provincial, regional, national benchmark prices  
- Engineered benchmark features  

### 4.5 Stratified Split (70/15/15)

- Stratify on price quantile bins  
- Prevent distribution shift  
- Ensures all price ranges represented  

### 4.6 Stage 3 — Final Feature Engineering (Fit/Transform)

- Missingness flags  
- Convert -1 → NaN  
- Final imputation (train-fitted)  
- price_per_m2, log(area)  
- Geo aggregates  
- Target encoding (train-only)  

---

## 5. Machine Learning Workflow

### Model Modes

**Linear Regression**  
- Numeric only  
- No TE  
- No derived leakage features  

**Random Forest / XGBoost**  
- Numeric + TE  
- All engineered features  
- Drop leak-prone features  

---

## 6. Model Performance Summary

**Linear Regression**  
- R²: 0.58–0.60  
- Underfits  

**Random Forest (Best)**  
- R²: 0.72–0.76  
- MAE ~54k–62k  

**XGBoost**  
- Similar R², slightly higher MAE  

---

## 7. Why No Leakage Occurs

- All encoders, imputers, geofeatures fitted **only on train**  
- Validation/Test never influence feature engineering  
- Benchmarks and external merges contain no target info  

---

## 8. Running the Pipeline

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```python
from pipelines.pipeline_runner import run_full_pipeline_with_split
run_full_pipeline_with_split()
```

---

## 9. Future Improvements

- OSM distance-based features  
- Nearest-neighbor comparable prices  
- Micro-market clustering  
- CatBoost / LightGBM  
- Optuna tuning  

---

## 10. License

MIT (recommended)

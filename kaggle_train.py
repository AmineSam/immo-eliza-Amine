"""
Standalone Kaggle Training Script
This script is self-contained and can run on Kaggle with minimal dependencies.
It loads the pre-processed CSV and performs:
- Stage 3 feature engineering (inline)
- Train/test split (configurable)
- Model training and evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

# Data path (adjust for Kaggle)
DATA_PATH = "data_for_kaggle.csv"

# Split configuration
TEST_SIZE = 0.20  # 80/20 split (change to 0.15 for 85/15, 0.10 for 90/10)
RANDOM_STATE = 42

# Model configuration
N_ESTIMATORS = 100
MAX_DEPTH = 20
MIN_SAMPLES_SPLIT = 5

# ============================================================
# STAGE 3: FEATURE ENGINEERING (INLINE)
# ============================================================

# Constants
MISSINGNESS_NUMERIC_COLS = [
    "area", "state", "facades_number", "is_furnished", "has_terrace", "has_garden",
    "has_swimming_pool", "has_equipped_kitchen", "build_year", "cellar",
    "has_garage", "bathrooms", "heating_type", "sewer_connection",
    "certification_electrical_installation", "preemption_right", "flooding_area_type",
    "leased", "living_room_surface", "attic_house", "glazing_type",
    "elevator", "access_disabled", "toilets", "cadastral_income_house",
]

LOG_FEATURES = ["area"]

TARGET_ENCODING_COLS = ["property_subtype", "property_type", "postal_code", "locality"]
TARGET_ENCODING_ALPHA = 100.0


def add_missingness_flags(df):
    """Add *_missing flags for columns with -1 as missing indicator."""
    df = df.copy()
    for col in MISSINGNESS_NUMERIC_COLS:
        if col in df.columns:
            df[f"{col}_missing"] = (df[col] == -1).astype(int)
    return df


def convert_minus1_to_nan(df):
    """Convert -1 to NaN for numeric/coded columns."""
    df = df.copy()
    for col in MISSINGNESS_NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].replace(-1, np.nan)
    return df


def add_log_features(df):
    """Add log-transformed features."""
    df = df.copy()
    for col in LOG_FEATURES:
        if col in df.columns:
            col_vals = df[col]
            mask = col_vals > 0
            log_col = np.full(len(df), np.nan)
            log_col[mask] = np.log1p(col_vals[mask])
            df[f"{col}_log"] = log_col
    return df


def target_encode_train(df_train, col, target="price", alpha=TARGET_ENCODING_ALPHA):
    """Fit target encoding on training data."""
    global_mean = df_train[target].mean()
    stats = df_train.groupby(col)[target].agg(["mean", "count"])
    smoothed = (stats["count"] * stats["mean"] + alpha * global_mean) / (stats["count"] + alpha)
    return smoothed.to_dict(), global_mean


def apply_target_encoding(df, col, mapping, global_mean):
    """Apply pre-fitted target encoding."""
    df = df.copy()
    out_col = f"{col}_te_price"
    df[out_col] = df[col].map(mapping).fillna(global_mean)
    return df


def impute_features(df, numeric_medians, ordinal_modes):
    """Apply imputation using pre-fitted statistics."""
    df = df.copy()
    
    # Numeric → median
    for col, med in numeric_medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(med)
    
    # Binary → 0
    binary_cols = [
        "cellar", "has_garage", "has_swimming_pool", "has_equipped_kitchen",
        "access_disabled", "elevator", "leased", "is_furnished",
        "has_terrace", "has_garden"
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Ordinal → mode
    for col, mode in ordinal_modes.items():
        if col in df.columns:
            df[col] = df[col].fillna(mode)
    
    return df


def fit_stage3(df_train):
    """Fit Stage 3 transformations on training data."""
    df = df_train.copy()
    
    # 1) Missingness
    df = add_missingness_flags(df)
    df = convert_minus1_to_nan(df)
    
    # 2) Fit imputers (early)
    numeric_cont = [
        "area", "rooms", "living_room_surface", "build_year",
        "facades_number", "bathrooms", "toilets", "cadastral_income_house",
        "median_income", "apt_avg_m2_province", "house_avg_m2_province",
        "apt_avg_m2_region", "house_avg_m2_region",
        "province_benchmark_m2", "region_benchmark_m2", "national_benchmark_m2",
    ]
    
    ordinal_cols = [
        "heating_type", "glazing_type", "sewer_connection",
        "certification_electrical_installation", "preemption_right",
        "flooding_area_type", "attic_house", "state", "region", "province",
    ]
    
    numeric_medians = {}
    for col in numeric_cont:
        if col in df.columns:
            numeric_medians[col] = df[col].median()
    
    ordinal_modes = {}
    for col in ordinal_cols:
        if col in df.columns:
            mode = df[col].mode(dropna=True)
            ordinal_modes[col] = mode.iloc[0] if len(mode) > 0 else 0
    
    df = impute_features(df, numeric_medians, ordinal_modes)
    
    # 3) Core FE
    df = add_log_features(df)
    
    # 4) Fit target encoding
    te_maps = {}
    global_means = {}
    for col in TARGET_ENCODING_COLS:
        if col in df.columns:
            te_maps[col], global_means[col] = target_encode_train(df, col)
    
    # 5) Refit imputers after FE
    for col in numeric_cont + ["area_log"]:
        if col in df.columns:
            numeric_medians[col] = df[col].median()
    
    return {
        "numeric_medians": numeric_medians,
        "ordinal_modes": ordinal_modes,
        "te_maps": te_maps,
        "global_means": global_means,
    }


def transform_stage3(df, fitted_params):
    """Transform data using fitted Stage 3 parameters."""
    df = df.copy()
    
    # 1) Missingness
    df = add_missingness_flags(df)
    df = convert_minus1_to_nan(df)
    
    # 2) Early imputation
    df = impute_features(df, fitted_params["numeric_medians"], fitted_params["ordinal_modes"])
    
    # 3) Core FE
    df = add_log_features(df)
    
    # 4) Apply target encoding
    for col, mapping in fitted_params["te_maps"].items():
        if col in df.columns:
            df = apply_target_encoding(df, col, mapping, fitted_params["global_means"][col])
    
    # 5) Final imputation
    df = impute_features(df, fitted_params["numeric_medians"], fitted_params["ordinal_modes"])
    
    return df


def prepare_X_y(df, model_type="rf"):
    """Prepare features and target for modeling."""
    df = df.copy()
    
    # Drop technical columns
    drop_cols = ["property_id", "url", "address"]
    drop_cols += [c for c in df.columns if c.startswith("__stage")]
    df = df.drop(columns=drop_cols, errors="ignore")
    
    # Separate target
    if "price" not in df.columns:
        raise ValueError("Column 'price' not found")
    y = df["price"]
    X = df.drop(columns=["price"])
    
    # Remove leakage features
    leakage_cols = ["price_log"]
    leakage_cols += [c for c in X.columns if c.startswith("diff_to_") or c.startswith("ratio_to_")]
    X = X.drop(columns=leakage_cols, errors="ignore")
    
    # Model-specific filtering
    if model_type in ("rf", "xgb"):
        # For trees: numeric only, TE is useful
        X = X.select_dtypes(include=[np.number])
        X = X.drop(columns=[c for c in X.columns if c.endswith("_log")], errors="ignore")
    
    return X, y


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    print("=" * 60)
    print("KAGGLE TRAINING SCRIPT")
    print("=" * 60)
    
    # Load data
    print(f"\n[LOAD] Reading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"   ✓ Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # Drop technical columns before split
    df_to_split = df.drop(
        columns=["url", "property_id", "address"],
        errors="ignore"
    )
    
    # Split data
    print(f"\n[SPLIT] Train/Test split ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)})...")
    df_train, df_test = train_test_split(
        df_to_split,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    print(f"   ✓ Train: {df_train.shape[0]:,} rows")
    print(f"   ✓ Test:  {df_test.shape[0]:,} rows")
    
    # Stage 3: Feature engineering
    print("\n[STAGE 3] Fitting feature engineering on training data...")
    fitted_params = fit_stage3(df_train)
    print("   ✓ Fitted Stage 3 transformations")
    
    print("\n[STAGE 3] Transforming train and test sets...")
    df_train_transformed = transform_stage3(df_train, fitted_params)
    df_test_transformed = transform_stage3(df_test, fitted_params)
    print(f"   ✓ Train transformed: {df_train_transformed.shape[1]} features")
    print(f"   ✓ Test transformed:  {df_test_transformed.shape[1]} features")
    
    # Prepare X, y
    print("\n[PREPARE] Preparing features and target...")
    X_train, y_train = prepare_X_y(df_train_transformed, model_type="rf")
    X_test, y_test = prepare_X_y(df_test_transformed, model_type="rf")
    print(f"   ✓ X_train: {X_train.shape}")
    print(f"   ✓ X_test:  {X_test.shape}")
    
    # Train model
    print(f"\n[TRAIN] Training Random Forest (n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH})...")
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    model.fit(X_train, y_train)
    print("   ✓ Model trained")
    
    # Evaluate
    print("\n[EVALUATE] Evaluating model...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nTrain Set:")
    print(f"  MAE: {train_mae:,.2f}")
    print(f"  R²:  {train_r2:.4f}")
    print(f"\nTest Set:")
    print(f"  MAE: {test_mae:,.2f}")
    print(f"  R²:  {test_r2:.4f}")
    print("\n" + "=" * 60)
    
    # Feature importance (top 20)
    print("\nTop 20 Feature Importances:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    for idx, row in feature_importance.iterrows():
        print(f"  {row['feature']:40s} {row['importance']:.4f}")
    
    return model, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    model, X_train, X_test, y_train, y_test = main()

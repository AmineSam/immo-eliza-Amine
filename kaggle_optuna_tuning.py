"""
Kaggle Hyperparameter Tuning Script
Optimizes XGBoost, LightGBM, and CatBoost with Optuna
Saves models at each stage as .pkl files
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "/kaggle/input/your-dataset/data_for_kaggle.csv"  # Update this path
OUTPUT_DIR = "/kaggle/working/"  # Kaggle output directory
TEST_SIZE = 0.15
RANDOM_STATE = 42
N_TRIALS = 100  # Reduce to 50 if time is limited

# ============================================================
# STAGE 3 FEATURE ENGINEERING (INLINE)
# ============================================================

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
    df = df.copy()
    for col in MISSINGNESS_NUMERIC_COLS:
        if col in df.columns:
            df[f"{col}_missing"] = (df[col] == -1).astype(int)
    return df


def convert_minus1_to_nan(df):
    df = df.copy()
    for col in MISSINGNESS_NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].replace(-1, np.nan)
    return df


def add_log_features(df):
    df = df.copy()
    for col in LOG_FEATURES:
        if col in df.columns:
            col_vals = df[col]
            mask = col_vals > 0
            log_col = np.full(len(df), np.nan)
            log_col[mask] = np.log1p(col_vals[mask])
            df[f"{col}_log"] = log_col
    return df


def impute_features(df, numeric_medians, ordinal_modes):
    df = df.copy()
    for col, med in numeric_medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(med)
    
    binary_cols = [
        "cellar", "has_garage", "has_swimming_pool", "has_equipped_kitchen",
        "access_disabled", "elevator", "leased", "is_furnished",
        "has_terrace", "has_garden"
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    for col, mode in ordinal_modes.items():
        if col in df.columns:
            df[col] = df[col].fillna(mode)
    
    return df


def fit_stage3(df_train):
    df = df_train.copy()
    df = add_missingness_flags(df)
    df = convert_minus1_to_nan(df)
    
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
    df = add_log_features(df)
    
    te_maps = {}
    global_means = {}
    for col in TARGET_ENCODING_COLS:
        if col in df.columns:
            global_mean = df["price"].mean()
            stats = df.groupby(col)["price"].agg(["mean", "count"])
            smoothed = (stats["count"] * stats["mean"] + TARGET_ENCODING_ALPHA * global_mean) / (
                stats["count"] + TARGET_ENCODING_ALPHA
            )
            te_maps[col] = smoothed.to_dict()
            global_means[col] = global_mean
    
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
    df = df.copy()
    df = add_missingness_flags(df)
    df = convert_minus1_to_nan(df)
    df = impute_features(df, fitted_params["numeric_medians"], fitted_params["ordinal_modes"])
    df = add_log_features(df)
    
    for col, mapping in fitted_params["te_maps"].items():
        if col in df.columns:
            out_col = f"{col}_te_price"
            df[out_col] = df[col].map(mapping).fillna(fitted_params["global_means"][col])
    
    df = impute_features(df, fitted_params["numeric_medians"], fitted_params["ordinal_modes"])
    return df


def prepare_X_y(df):
    df = df.copy()
    drop_cols = ["property_id", "url", "address"]
    drop_cols += [c for c in df.columns if c.startswith("__stage")]
    df = df.drop(columns=drop_cols, errors="ignore")
    
    y = df["price"]
    X = df.drop(columns=["price"])
    
    leakage_cols = ["price_log"]
    leakage_cols += [c for c in X.columns if c.startswith("diff_to_") or c.startswith("ratio_to_")]
    X = X.drop(columns=leakage_cols, errors="ignore")
    X = X.select_dtypes(include=[np.number])
    X = X.drop(columns=[c for c in X.columns if c.endswith("_log")], errors="ignore")
    
    return X, y


# ============================================================
# OPTUNA OBJECTIVES
# ============================================================

def objective_xgboost(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 800),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
        'gamma': trial.suggest_float('gamma', 0, 3),
        'reg_alpha': trial.suggest_float('reg_alpha', 1, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 5),
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'tree_method': 'hist',
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    y_pred = model.predict(X_val)
    val_r2 = r2_score(y_val, y_pred)
    
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    overfitting_penalty = max(0, (train_r2 - val_r2) - 0.05) * 2
    
    return val_r2 - overfitting_penalty


def objective_lightgbm(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 800),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 50),
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 1, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 5),
        'min_split_gain': trial.suggest_float('min_split_gain', 0, 2),
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1,
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    
    y_pred = model.predict(X_val)
    val_r2 = r2_score(y_val, y_pred)
    
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    overfitting_penalty = max(0, (train_r2 - val_r2) - 0.05) * 2
    
    return val_r2 - overfitting_penalty


def objective_catboost(trial, X_train, y_train, X_val, y_val):
    params = {
        'iterations': trial.suggest_int('iterations', 300, 800),
        'depth': trial.suggest_int('depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_strength': trial.suggest_float('random_strength', 1, 5),
        'border_count': trial.suggest_int('border_count', 128, 254),
        'random_state': RANDOM_STATE,
        'verbose': False,
        'thread_count': -1,
    }
    
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    
    y_pred = model.predict(X_val)
    val_r2 = r2_score(y_val, y_pred)
    
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    overfitting_penalty = max(0, (train_r2 - val_r2) - 0.05) * 2
    
    return val_r2 - overfitting_penalty


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 80)
    print("KAGGLE HYPERPARAMETER TUNING WITH MODEL SAVING")
    print("=" * 80)
    
    # Load data
    print(f"\n[1/7] Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"   ✓ Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # Split
    df_to_split = df.drop(columns=["url", "property_id", "address"], errors="ignore")
    df_train_val, df_test = train_test_split(df_to_split, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    df_train, df_val = train_test_split(df_train_val, test_size=0.15, random_state=RANDOM_STATE)
    
    print(f"\n[2/7] Data split:")
    print(f"   ✓ Train: {df_train.shape[0]:,}")
    print(f"   ✓ Val:   {df_val.shape[0]:,}")
    print(f"   ✓ Test:  {df_test.shape[0]:,}")
    
    # Feature engineering
    print(f"\n[3/7] Feature engineering...")
    fitted_params = fit_stage3(df_train)
    
    # Save fitted parameters
    with open(f"{OUTPUT_DIR}stage3_fitted_params.pkl", 'wb') as f:
        pickle.dump(fitted_params, f)
    print(f"   ✓ Saved: stage3_fitted_params.pkl")
    
    df_train_transformed = transform_stage3(df_train, fitted_params)
    df_val_transformed = transform_stage3(df_val, fitted_params)
    df_test_transformed = transform_stage3(df_test, fitted_params)
    
    X_train, y_train = prepare_X_y(df_train_transformed)
    X_val, y_val = prepare_X_y(df_val_transformed)
    X_test, y_test = prepare_X_y(df_test_transformed)
    
    print(f"   ✓ Features: {X_train.shape[1]}")
    
    # Hyperparameter tuning
    print(f"\n[4/7] Hyperparameter tuning ({N_TRIALS} trials per model)...")
    
    # XGBoost
    print("\n   [XGBoost] Optimizing...")
    study_xgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE))
    study_xgb.optimize(lambda trial: objective_xgboost(trial, X_train, y_train, X_val, y_val), 
                       n_trials=N_TRIALS, show_progress_bar=True)
    best_xgb_params = study_xgb.best_params
    print(f"   ✓ Best XGBoost R²: {study_xgb.best_value:.4f}")
    
    # Save study
    with open(f"{OUTPUT_DIR}study_xgb.pkl", 'wb') as f:
        pickle.dump(study_xgb, f)
    
    # LightGBM
    print("\n   [LightGBM] Optimizing...")
    study_lgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE))
    study_lgb.optimize(lambda trial: objective_lightgbm(trial, X_train, y_train, X_val, y_val), 
                       n_trials=N_TRIALS, show_progress_bar=True)
    best_lgb_params = study_lgb.best_params
    print(f"   ✓ Best LightGBM R²: {study_lgb.best_value:.4f}")
    
    with open(f"{OUTPUT_DIR}study_lgb.pkl", 'wb') as f:
        pickle.dump(study_lgb, f)
    
    # CatBoost
    print("\n   [CatBoost] Optimizing...")
    study_cat = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE))
    study_cat.optimize(lambda trial: objective_catboost(trial, X_train, y_train, X_val, y_val), 
                       n_trials=N_TRIALS, show_progress_bar=True)
    best_cat_params = study_cat.best_params
    print(f"   ✓ Best CatBoost R²: {study_cat.best_value:.4f}")
    
    with open(f"{OUTPUT_DIR}study_cat.pkl", 'wb') as f:
        pickle.dump(study_cat, f)
    
    # Train final models
    print(f"\n[5/7] Training final models...")
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    
    model_xgb = xgb.XGBRegressor(**best_xgb_params, random_state=RANDOM_STATE, n_jobs=-1)
    model_lgb = lgb.LGBMRegressor(**best_lgb_params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    model_cat = CatBoostRegressor(**best_cat_params, random_state=RANDOM_STATE, verbose=False)
    
    model_xgb.fit(X_train_full, y_train_full)
    model_lgb.fit(X_train_full, y_train_full)
    model_cat.fit(X_train_full, y_train_full)
    
    # Save individual models
    with open(f"{OUTPUT_DIR}model_xgb.pkl", 'wb') as f:
        pickle.dump(model_xgb, f)
    with open(f"{OUTPUT_DIR}model_lgb.pkl", 'wb') as f:
        pickle.dump(model_lgb, f)
    with open(f"{OUTPUT_DIR}model_cat.pkl", 'wb') as f:
        pickle.dump(model_cat, f)
    
    print("   ✓ Saved: model_xgb.pkl, model_lgb.pkl, model_cat.pkl")
    
    # Ensemble
    print(f"\n[6/7] Creating ensemble...")
    estimators = [
        ('xgb', model_xgb),
        ('lgb', model_lgb),
        ('cat', model_cat)
    ]
    
    ensemble = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=10),
        cv=5,
        n_jobs=-1
    )
    ensemble.fit(X_train_full, y_train_full)
    
    # Save ensemble
    with open(f"{OUTPUT_DIR}model_ensemble.pkl", 'wb') as f:
        pickle.dump(ensemble, f)
    
    print("   ✓ Saved: model_ensemble.pkl")
    
    # Evaluation
    print(f"\n[7/7] Evaluation...")
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    models = {
        'XGBoost': model_xgb,
        'LightGBM': model_lgb,
        'CatBoost': model_cat,
        'Ensemble': ensemble
    }
    
    results = []
    for name, model in models.items():
        y_train_pred = model.predict(X_train_full)
        y_test_pred = model.predict(X_test)
        
        train_r2 = r2_score(y_train_full, y_train_pred)
        train_mae = mean_absolute_error(y_train_full, y_train_pred)
        
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        overfitting = train_r2 - test_r2
        target_met = test_r2 > 0.9 and overfitting < 0.1
        
        results.append({
            'Model': name,
            'Train_R2': train_r2,
            'Test_R2': test_r2,
            'Train_MAE': train_mae,
            'Test_MAE': test_mae,
            'Overfitting': overfitting,
            'Target_Met': target_met
        })
        
        print(f"\n{name}:")
        print(f"  Train - R²: {train_r2:.4f}, MAE: {train_mae:,.0f}")
        print(f"  Test  - R²: {test_r2:.4f}, MAE: {test_mae:,.0f}")
        print(f"  Overfitting: {overfitting:.4f} {'✓' if overfitting < 0.1 else '✗'}")
        print(f"  Target Met: {'✓' if target_met else '✗'}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUTPUT_DIR}model_results.csv", index=False)
    print(f"\n   ✓ Saved: model_results.csv")
    
    # Save best parameters
    params_summary = {
        'xgboost': best_xgb_params,
        'lightgbm': best_lgb_params,
        'catboost': best_cat_params
    }
    with open(f"{OUTPUT_DIR}best_params.pkl", 'wb') as f:
        pickle.dump(params_summary, f)
    
    print(f"   ✓ Saved: best_params.pkl")
    
    print("\n" + "=" * 80)
    print("ALL FILES SAVED TO:", OUTPUT_DIR)
    print("=" * 80)
    print("\nSaved files:")
    print("  - stage3_fitted_params.pkl (feature engineering parameters)")
    print("  - study_xgb.pkl, study_lgb.pkl, study_cat.pkl (Optuna studies)")
    print("  - model_xgb.pkl, model_lgb.pkl, model_cat.pkl (individual models)")
    print("  - model_ensemble.pkl (stacking ensemble)")
    print("  - best_params.pkl (best hyperparameters)")
    print("  - model_results.csv (performance metrics)")
    
    return ensemble, models, results_df


if __name__ == "__main__":
    ensemble, models, results = main()

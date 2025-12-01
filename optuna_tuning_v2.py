# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-28T22:11:28.545419Z","iopub.execute_input":"2025-11-28T22:11:28.545665Z","iopub.status.idle":"2025-11-28T22:15:30.213934Z","shell.execute_reply.started":"2025-11-28T22:11:28.545638Z","shell.execute_reply":"2025-11-28T22:15:30.212829Z"}}
"""
FAST GPU MODEL TUNING — XGBoost + CatBoost ONLY
- Optuna + pruning
- Early stopping
- Overfitting penalty (gap - 0.14)
- Saves each model with joblib
- NO SHAP
- NO ENSEMBLE
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

import xgboost as xgb
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings('ignore')

# ========================================
# CONFIG
# ========================================

DATA_PATH = "/kaggle/input/data-for-kaggle/data_for_kaggle.csv"
OUTPUT_DIR = "/kaggle/working/"

TEST_SIZE = 0.15
RANDOM_STATE = 42
N_TRIALS = 200      # reduced for speed; can increase
EARLY_STOPPING_ROUNDS = 50

TARGET_ENCODING_ALPHA = 100.0

# ========================================
# FEATURE ENGINEERING (SAME AS BEFORE)
# ========================================

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

GEO_COLUMNS = [
    "apt_avg_m2_province", "house_avg_m2_province",
    "apt_avg_m2_region", "house_avg_m2_region",
    "province_benchmark_m2", "region_benchmark_m2",
    "national_benchmark_m2"
]


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
            vals = df[col]
            mask = vals > 0
            out = np.full(len(df), np.nan)
            out[mask] = np.log1p(vals[mask])
            df[f"{col}_log"] = out
    return df


def impute_features(df, numeric_medians, ordinal_modes):
    df = df.copy()

    # Numeric
    for col, med in numeric_medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(med)

    # Binary
    binary_cols = [
        "cellar", "has_garage", "has_swimming_pool", "has_equipped_kitchen",
        "access_disabled", "elevator", "leased", "is_furnished",
        "has_terrace", "has_garden"
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Ordinal
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
        "facades_number", "bathrooms", "toilets",
        "cadastral_income_house", "median_income"
    ]

    ordinal_cols = [
        "heating_type", "glazing_type", "sewer_connection",
        "certification_electrical_installation", "preemption_right",
        "flooding_area_type", "attic_house", "state",
        "region", "province"
    ]

    numeric_medians = {}
    for col in numeric_cont:
        if col in df.columns:
            numeric_medians[col] = df[col].median()

    ordinal_modes = {}
    for col in ordinal_cols:
        if col in df.columns:
            mode = df[col].mode(dropna=True)
            ordinal_modes[col] = mode.iloc[0] if len(mode) else 0

    df = impute_features(df, numeric_medians, ordinal_modes)
    df = add_log_features(df)

    te_maps = {}
    global_means = {}

    for col in TARGET_ENCODING_COLS:
        if col in df.columns:
            global_mean = df["price"].mean()
            stats = df.groupby(col)["price"].agg(["mean", "count"])
            smoothed = (
                (stats["count"] * stats["mean"] + TARGET_ENCODING_ALPHA * global_mean)
                / (stats["count"] + TARGET_ENCODING_ALPHA)
            )
            te_maps[col] = smoothed.to_dict()
            global_means[col] = global_mean

    if "area_log" in df.columns:
        numeric_medians["area_log"] = df["area_log"].median()

    return {
        "numeric_medians": numeric_medians,
        "ordinal_modes": ordinal_modes,
        "te_maps": te_maps,
        "global_means": global_means,
    }


def transform_stage3(df, fitted):
    df = df.copy()
    df = add_missingness_flags(df)
    df = convert_minus1_to_nan(df)

    # backup geo fields
    geo_backup = {col: df[col].copy() for col in GEO_COLUMNS if col in df.columns}

    df = impute_features(df, fitted["numeric_medians"], fitted["ordinal_modes"])

    # restore geo fields
    for col, vals in geo_backup.items():
        df[col] = vals

    df = add_log_features(df)

    for col, mapping in fitted["te_maps"].items():
        if col in df.columns:
            df[f"{col}_te_price"] = df[col].map(mapping).fillna(fitted["global_means"][col])

    df = impute_features(df, fitted["numeric_medians"], fitted["ordinal_modes"])

    # restore geo again
    for col, vals in geo_backup.items():
        df[col] = vals

    return df


def prepare_X_y(df):
    df = df.copy()

    drop_cols = ["property_id", "url", "address"]
    drop_cols += [c for c in df.columns if c.startswith("__stage")]
    df = df.drop(columns=drop_cols, errors="ignore")

    y = df["price"]
    X = df.drop(columns=["price"], errors="ignore")

    leakage_cols = ["price_log"]
    leakage_cols += [c for c in X.columns if c.startswith("diff_to_") or c.startswith("ratio_to_")]
    X = X.drop(columns=leakage_cols, errors="ignore")

    X = X.select_dtypes(include=[np.number])

    X = X.drop(columns=[c for c in X.columns if c.endswith("_log")], errors="ignore")

    return X, y

# ========================================
# PENALTY TO LIMIT OVERFITTING
# ========================================

def penalty(train_r2, val_r2):
    gap = train_r2 - val_r2
    return max(0, gap - 0.14)


# ========================================
# OPTUNA OBJECTIVES (SUPER OPTIMIZED)
# ========================================

def objective_xgb(trial, X_train, y_train, X_val, y_val):

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 900),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        # GPU
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'random_state': RANDOM_STATE,
    }

    model = xgb.XGBRegressor(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=False
    )

    val_pred = model.predict(X_val)
    train_pred = model.predict(X_train)

    val_r2 = r2_score(y_val, val_pred)
    train_r2 = r2_score(y_train, train_pred)

    # Pruning for Optuna
    trial.report(val_r2, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_r2 - penalty(train_r2, val_r2)


def objective_cat(trial, X_train, y_train, X_val, y_val):

    params = {
        'iterations': trial.suggest_int('iterations', 300, 900),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 1, 5),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'border_count': trial.suggest_int('border_count', 128, 254),
        # GPU
        'task_type': 'GPU',
        'devices': '0',
        'random_state': RANDOM_STATE,
        'verbose': False
    }

    model = CatBoostRegressor(**params)

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=False
    )

    val_pred = model.predict(X_val)
    train_pred = model.predict(X_train)

    val_r2 = r2_score(y_val, val_pred)
    train_r2 = r2_score(y_train, train_pred)

    # Pruning
    trial.report(val_r2, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_r2 - penalty(train_r2, val_r2)


# ========================================
# MAIN FLOW
# ========================================

def main():
    print("=" * 70)
    print("FAST GPU TUNING — XGBOOST + CATBOOST ONLY (NO SHAP, NO ENSEMBLE)")
    print("=" * 70)

    # load data
    df = pd.read_csv(DATA_PATH)
    df2 = df.drop(columns=["url", "address"], errors="ignore")

    # split
    df_train_val, df_test = train_test_split(df2, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    df_train, df_val = train_test_split(df_train_val, test_size=0.15, random_state=RANDOM_STATE)

    # stage 3
    fitted = fit_stage3(df_train)
    df_train_t = transform_stage3(df_train, fitted)
    df_val_t   = transform_stage3(df_val, fitted)
    df_test_t  = transform_stage3(df_test, fitted)

    X_train, y_train = prepare_X_y(df_train_t)
    X_val,   y_val   = prepare_X_y(df_val_t)
    X_test,  y_test  = prepare_X_y(df_test_t)

    # =============================
    # OPTUNA — XGBoost
    # =============================
    print("\n=== OPTUNA: XGBoost ===")

    study_xgb = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
        pruner=MedianPruner()
    )

    study_xgb.optimize(
        lambda t: objective_xgb(t, X_train, y_train, X_val, y_val),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )

    best_xgb = study_xgb.best_params
    print("Best XGB params:", best_xgb)

    # train final XGB
    model_xgb = xgb.XGBRegressor(
        **best_xgb,
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        random_state=RANDOM_STATE
    )
    model_xgb.fit(X_train, y_train)
    joblib.dump(model_xgb, f"{OUTPUT_DIR}/best_xgb_model.joblib")


    # =============================
    # OPTUNA — CatBoost
    # =============================
    print("\n=== OPTUNA: CatBoost ===")

    study_cat = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
        pruner=MedianPruner()
    )

    study_cat.optimize(
        lambda t: objective_cat(t, X_train, y_train, X_val, y_val),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )

    best_cat = study_cat.best_params
    print("Best CAT params:", best_cat)

    # train final CAT
    model_cat = CatBoostRegressor(
        **best_cat,
        task_type="GPU",
        devices="0",
        verbose=False,
        random_state=RANDOM_STATE
    )
    model_cat.fit(X_train, y_train)
    joblib.dump(model_cat, f"{OUTPUT_DIR}/best_cat_model.joblib")


    # =============================
    # FINAL EVALUATION
    # =============================

    def evaluate(name, model):
        pred_train = model.predict(X_train)
        pred_test  = model.predict(X_test)
        print(f"\n{name}:")
        print(" Train R²:", round(r2_score(y_train, pred_train), 4))
        print(" Test  R²:", round(r2_score(y_test, pred_test), 4))
        print(" Train-Test gap:", round(r2_score(y_train, pred_train) - r2_score(y_test, pred_test), 4))
        print(" MAE Test:", round(mean_absolute_error(y_test, pred_test), 0))

    evaluate("XGBoost", model_xgb)
    evaluate("CatBoost", model_cat)

    print("\nSaved best models in /kaggle/working/")
    print("Done.")

    return model_xgb, model_cat


if __name__ == "__main__":
    main()

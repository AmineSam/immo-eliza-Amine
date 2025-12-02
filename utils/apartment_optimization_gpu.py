"""
APARTMENT-ONLY FAST GPU TUNING — XGBoost + CatBoost
- Filters property_type == "apartment"
- Stage 3 preprocessing inline (same as general pipeline)
- Optuna on GPU (XGB + CAT)
- Objective: MINIMIZE validation MAE
- Reports R² and MAE for Train / Val / Test
- Weighted ensemble search (XGB vs CAT) based on Val MAE
- Does NOT save models (only writes summary .txt)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================

DATA_PATH = "/kaggle/input/data-for-kaggle/data_for_kaggle.csv"
OUTPUT_DIR = "/kaggle/working/"

TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42
N_TRIALS = 100
EARLY_STOPPING_ROUNDS = 50

TARGET_ENCODING_ALPHA = 100.0

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# =========================================================
# FEATURE ENGINEERING HELPERS (SAME AS ABOVE)
# =========================================================

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
        "facades_number", "bathrooms", "toilets",
        "cadastral_income_house", "median_income"
    ]

    ordinal_cols = [
        "heating_type", "glazing_type", "sewer_connection",
        "certification_electrical_installation", "preemption_right",
        "flooding_area_type", "attic_house", "state",
        "region", "province"
    ]

    numeric_medians = {
        col: df[col].median() for col in numeric_cont if col in df.columns
    }

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

    geo_backup = {col: df[col].copy() for col in GEO_COLUMNS if col in df.columns}

    df = impute_features(df, fitted["numeric_medians"], fitted["ordinal_modes"])

    for col, vals in geo_backup.items():
        df[col] = vals

    df = add_log_features(df)

    for col, mapping in fitted["te_maps"].items():
        if col in df.columns:
            df[f"{col}_te_price"] = df[col].map(mapping).fillna(
                fitted["global_means"][col]
            )

    df = impute_features(df, fitted["numeric_medians"], fitted["ordinal_modes"])

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


# =========================================================
# OPTUNA OBJECTIVES — MINIMIZE MAE
# =========================================================

def objective_xgb(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 600, 2200),
        "max_depth": trial.suggest_int("max_depth", 6, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.20, log=True),

        "subsample": trial.suggest_float("subsample", 0.7, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),

        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),

        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
        "random_state": RANDOM_STATE,
    }

    model = xgb.XGBRegressor(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=False,
    )

    val_pred = model.predict(X_val)
    train_pred = model.predict(X_train)

    val_mae = mean_absolute_error(y_val, val_pred)
    train_mae = mean_absolute_error(y_train, train_pred)

    val_r2 = r2_score(y_val, val_pred)
    train_r2 = r2_score(y_train, train_pred)

    trial.set_user_attr("val_mae", float(val_mae))
    trial.set_user_attr("train_mae", float(train_mae))
    trial.set_user_attr("val_r2", float(val_r2))
    trial.set_user_attr("train_r2", float(train_r2))

    trial.report(val_mae, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_mae


def objective_cat(trial, X_train, y_train, X_val, y_val):
    params = {
        "iterations": trial.suggest_int("iterations", 800, 2200),
        "depth": trial.suggest_int("depth", 6, 11),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.20, log=True),

        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 8.0),
        "random_strength": trial.suggest_float("random_strength", 0.5, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.2, 2.5),
        "border_count": trial.suggest_int("border_count", 64, 255),

        "loss_function": "RMSE",
        "task_type": "GPU",
        "devices": "0",
        "random_state": RANDOM_STATE,
        "verbose": False,
    }

    model = CatBoostRegressor(**params)

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
        verbose=False,
    )

    val_pred = model.predict(X_val)
    train_pred = model.predict(X_train)

    val_mae = mean_absolute_error(y_val, val_pred)
    train_mae = mean_absolute_error(y_train, train_pred)

    val_r2 = r2_score(y_val, val_pred)
    train_r2 = r2_score(y_train, train_pred)

    trial.set_user_attr("val_mae", float(val_mae))
    trial.set_user_attr("train_mae", float(train_mae))
    trial.set_user_attr("val_r2", float(val_r2))
    trial.set_user_attr("train_r2", float(train_r2))

    trial.report(val_mae, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_mae


# =========================================================
# METRICS / LOGGING
# =========================================================

def evaluate_model(name, model, X_train, y_train, X_val, y_val, X_test, y_test):
    lines = []

    pred_train = model.predict(X_train)
    pred_val   = model.predict(X_val)
    pred_test  = model.predict(X_test)

    train_r2 = r2_score(y_train, pred_train)
    val_r2   = r2_score(y_val,   pred_val)
    test_r2  = r2_score(y_test,  pred_test)

    train_mae = mean_absolute_error(y_train, pred_train)
    val_mae   = mean_absolute_error(y_val,   pred_val)
    test_mae  = mean_absolute_error(y_test,  pred_test)

    lines.append(f"\n{name} PERFORMANCE (APARTMENT):")
    lines.append(f" Train R²: {train_r2:.4f}")
    lines.append(f" Val   R²: {val_r2:.4f}")
    lines.append(f" Test  R²: {test_r2:.4f}")
    lines.append(f" Train–Val R² gap: {train_r2 - val_r2:.4f}")
    lines.append(f" Train MAE: {train_mae:,.0f}")
    lines.append(f" Val   MAE: {val_mae:,.0f}")
    lines.append(f" Test  MAE: {test_mae:,.0f}")

    print("\n".join(lines))
    return lines


# =========================================================
# MAIN — APARTMENTS ONLY
# =========================================================

def main():
    log_lines = []
    log_lines.append("APARTMENT-ONLY FAST GPU TUNING — XGB + CAT")
    log_lines.append("=" * 70)

    print("=" * 70)
    print("APARTMENT-ONLY FAST GPU TUNING — XGBoost + CatBoost")
    print("=" * 70)

    df_full = pd.read_csv(DATA_PATH)
    df = df_full[df_full["property_type"] == "Apartment"].copy()

    log_lines.append(f"Total rows (all): {len(df_full)}")
    log_lines.append(f"Rows after filtering property_type == 'Apartment': {len(df)}")

    df = df.drop(columns=["url", "address"], errors="ignore")

    df_train_val, df_test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    df_train, df_val = train_test_split(
        df_train_val, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    fitted = fit_stage3(df_train)
    df_train_t = transform_stage3(df_train, fitted)
    df_val_t   = transform_stage3(df_val,   fitted)
    df_test_t  = transform_stage3(df_test,  fitted)

    X_train, y_train = prepare_X_y(df_train_t)
    X_val,   y_val   = prepare_X_y(df_val_t)
    X_test,  y_test  = prepare_X_y(df_test_t)

    log_lines.append(f"X_train shape: {X_train.shape}")
    log_lines.append(f"X_val   shape: {X_val.shape}")
    log_lines.append(f"X_test  shape: {X_test.shape}")

    # -----------------------------
    # XGB OPTUNA
    # -----------------------------
    print("\n=== OPTUNA: XGBoost (APARTMENT) — objective = Val MAE ===")
    log_lines.append("\n=== OPTUNA: XGBoost (APARTMENT) — objective = Val MAE ===")

    study_xgb = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=RANDOM_STATE),
        pruner=MedianPruner()
    )

    study_xgb.optimize(
        lambda t: objective_xgb(t, X_train, y_train, X_val, y_val),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_xgb_params = study_xgb.best_params
    best_xgb_val_mae = study_xgb.best_value

    print("\nBest XGB params (APARTMENT):")
    print(best_xgb_params)
    print(f"Best XGB validation MAE: {best_xgb_val_mae:,.0f}")

    log_lines.append("\nBest XGB params (APARTMENT):")
    log_lines.append(str(best_xgb_params))
    log_lines.append(f"Best XGB validation MAE: {best_xgb_val_mae:,.0f}")

    final_xgb_params = best_xgb_params.copy()
    final_xgb_params.update({
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
        "random_state": RANDOM_STATE,
    })
    model_xgb = xgb.XGBRegressor(**final_xgb_params)
    model_xgb.fit(X_train, y_train)

    # -----------------------------
    # CAT OPTUNA
    # -----------------------------
    print("\n=== OPTUNA: CatBoost (APARTMENT) — objective = Val MAE ===")
    log_lines.append("\n=== OPTUNA: CatBoost (APARTMENT) — objective = Val MAE ===")

    study_cat = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=RANDOM_STATE),
        pruner=MedianPruner()
    )

    study_cat.optimize(
        lambda t: objective_cat(t, X_train, y_train, X_val, y_val),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_cat_params = study_cat.best_params
    best_cat_val_mae = study_cat.best_value

    print("\nBest CAT params (APARTMENT):")
    print(best_cat_params)
    print(f"Best CAT validation MAE: {best_cat_val_mae:,.0f}")

    log_lines.append("\nBest CAT params (APARTMENT):")
    log_lines.append(str(best_cat_params))
    log_lines.append(f"Best CAT validation MAE: {best_cat_val_mae:,.0f}")

    final_cat_params = best_cat_params.copy()
    final_cat_params.update({
        "loss_function": "RMSE",
        "task_type": "GPU",
        "devices": "0",
        "random_state": RANDOM_STATE,
        "verbose": False,
    })
    model_cat = CatBoostRegressor(**final_cat_params)
    model_cat.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
        verbose=False,
    )

    # -----------------------------
    # BASE MODEL EVAL
    # -----------------------------
    xgb_lines = evaluate_model(
        "XGBoost", model_xgb, X_train, y_train, X_val, y_val, X_test, y_test
    )
    cat_lines = evaluate_model(
        "CatBoost", model_cat, X_train, y_train, X_val, y_val, X_test, y_test
    )
    log_lines.extend(xgb_lines)
    log_lines.extend(cat_lines)

    # -----------------------------
    # ENSEMBLE (MAE-based)
    # -----------------------------
    print("\n=== WEIGHTED ENSEMBLE SEARCH (APARTMENT, objective = Val MAE) ===")
    log_lines.append("\n=== WEIGHTED ENSEMBLE SEARCH (APARTMENT, objective = Val MAE) ===")

    xgb_train_pred = model_xgb.predict(X_train)
    xgb_val_pred   = model_xgb.predict(X_val)
    xgb_test_pred  = model_xgb.predict(X_test)

    cat_train_pred = model_cat.predict(X_train)
    cat_val_pred   = model_cat.predict(X_val)
    cat_test_pred  = model_cat.predict(X_test)

    best_w = None
    best_val_mae = np.inf
    best_summary = {}

    for w in np.linspace(0.0, 1.0, 21):
        train_ens = w * xgb_train_pred + (1 - w) * cat_train_pred
        val_ens   = w * xgb_val_pred   + (1 - w) * cat_val_pred
        test_ens  = w * xgb_test_pred  + (1 - w) * cat_test_pred

        train_mae = mean_absolute_error(y_train, train_ens)
        val_mae   = mean_absolute_error(y_val,   val_ens)
        test_mae  = mean_absolute_error(y_test,  test_ens)

        train_r2 = r2_score(y_train, train_ens)
        val_r2   = r2_score(y_val,   val_ens)
        test_r2  = r2_score(y_test,  test_ens)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_w = w
            best_summary = {
                "train_mae": train_mae,
                "val_mae": val_mae,
                "test_mae": test_mae,
                "train_r2": train_r2,
                "val_r2": val_r2,
                "test_r2": test_r2,
            }

    print(f"\nBest ensemble weight w (XGB share) = {best_w:.2f}")
    print(f" Ensemble Train MAE: {best_summary['train_mae']:,.0f}")
    print(f" Ensemble Val   MAE: {best_summary['val_mae']:,.0f}")
    print(f" Ensemble Test  MAE: {best_summary['test_mae']:,.0f}")
    print(f" Ensemble Train R²: {best_summary['train_r2']:.4f}")
    print(f" Ensemble Val   R²: {best_summary['val_r2']:.4f}")
    print(f" Ensemble Test  R²: {best_summary['test_r2']:.4f}")

    log_lines.append(f"\nBest ensemble weight w (XGB share) = {best_w:.2f}")
    log_lines.append(
        f" Ensemble Train MAE: {best_summary['train_mae']:,.0f} | "
        f"Val MAE: {best_summary['val_mae']:,.0f} | "
        f"Test MAE: {best_summary['test_mae']:,.0f}"
    )
    log_lines.append(
        f" Ensemble Train R²: {best_summary['train_r2']:.4f} | "
        f"Val R²: {best_summary['val_r2']:.4f} | "
        f"Test R²: {best_summary['test_r2']:.4f}"
    )

    # -----------------------------
    # WRITE SUMMARY
    # -----------------------------
    log_path = Path(OUTPUT_DIR) / "apartment_tuning_summary.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"\nSummary written to: {log_path}")
    print("\nDone (APARTMENT).")


if __name__ == "__main__":
    main()

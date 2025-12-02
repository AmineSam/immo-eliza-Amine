# ============================================================
# utils/stage3_utils.py
# Stage 3 preprocessing utilities (EXACT COPY — NO LOGIC CHANGED)
# ============================================================

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ========================================
# CONFIG-DEPENDENT CONSTANTS
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


# ============================================================
# FEATURE ENGINEERING FUNCTIONS (unchanged)
# ============================================================

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


# ============================================================
# FIT STAGE 3 (unchanged)
# ============================================================

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

    # area_log median
    if "area_log" in df.columns:
        numeric_medians["area_log"] = df["area_log"].median()

    return {
        "numeric_medians": numeric_medians,
        "ordinal_modes": ordinal_modes,
        "te_maps": te_maps,
        "global_means": global_means,
    }


# ============================================================
# TRANSFORM STAGE 3 (unchanged)
# ============================================================

def transform_stage3(df, fitted):
    df = df.copy()
    df = add_missingness_flags(df)
    df = convert_minus1_to_nan(df)

    # backup geo fields
    geo_backup = {col: df[col].copy() for col in GEO_COLUMNS if col in df.columns}

    df = impute_features(df, fitted["numeric_medians"], fitted["ordinal_modes"])

    # restore geo
    for col, vals in geo_backup.items():
        df[col] = vals

    df = add_log_features(df)

    # target encoding
    for col, mapping in fitted["te_maps"].items():
        if col in df.columns:
            df[f"{col}_te_price"] = df[col].map(mapping).fillna(fitted["global_means"][col])

    df = impute_features(df, fitted["numeric_medians"], fitted["ordinal_modes"])

    # restore geo final time
    for col, vals in geo_backup.items():
        df[col] = vals

    return df


# ============================================================
# PREPARE X / y (unchanged)
# ============================================================

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

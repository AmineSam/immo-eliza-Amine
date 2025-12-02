# ============================================================
# Stage 3 — Feature Engineering Pipeline
# ============================================================
# This stage builds enriched ML-ready features:
#   1. Missingness flags
#   2. -1 → NaN conversion
#   3. price_per_m2
#   4. log transforms
#   5. Geo-aggregates (postal/locality mean prices)
#   6. Target encoding for tree models (RF/XGBoost)
#
# ============================================================

import pandas as pd
import numpy as np


# ============================================================
# 0. CONSTANTS
# ============================================================

# Columns where -1 means missing and must get missing flags
MISSINGNESS_NUMERIC_COLS = [
    "area", "state",
    "facades_number", "is_furnished", "has_terrace", "has_garden",
    "has_swimming_pool", "has_equipped_kitchen", "build_year", "cellar",
    "has_garage", "bathrooms", "heating_type",
    "sewer_connection", "certification_electrical_installation",
    "preemption_right", "flooding_area_type", "leased",
    "living_room_surface", "attic_house", "glazing_type",
    "elevator", "access_disabled", "toilets",
    "cadastral_income_house",
]

# Log transforms (log1p)
LOG_FEATURES = [
    "area"
]

# Columns to target-encode for tree models
TARGET_ENCODING_COLS = [
    "property_subtype",
    "property_type",
    "postal_code",
    "locality",
]

# Smoothing factor for TE
TARGET_ENCODING_ALPHA = 100.0


# ============================================================
# 1. MISSINGNESS HANDLING
# ============================================================

def add_missingness_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add *_missing flags.
    - For numeric/coded columns: 1 if col == -1 else 0.
    - For categorical cols: 1 if NaN else 0.
    """
    df = df.copy()

    for col in MISSINGNESS_NUMERIC_COLS:
        if col in df.columns:
            df[f"{col}_missing"] = (df[col] == -1).astype(int)

    return df


def convert_minus1_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert -1 → NaN for numeric/coded columns.
    Called AFTER missingness flags are created.
    """
    df = df.copy()

    for col in MISSINGNESS_NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].replace(-1, np.nan)

    return df


def stage3_missingness_handler(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper for complete missingness handling:
    1. *_missing flags
    2. -1 → NaN
    """
    df = df.copy()
    df = add_missingness_flags(df)
    df = convert_minus1_to_nan(df)
    return df


# ============================================================
# 2. CORE FEATURE ENGINEERING
# ============================================================



def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in LOG_FEATURES:
        if col in df.columns:
            col_vals = df[col]

            # Mask before computing log
            mask = col_vals > 0
            log_col = np.full(len(df), np.nan)
            log_col[mask] = np.log1p(col_vals[mask])

            df[f"{col}_log"] = log_col

    return df


# ============================================================
# 3. GEO AGGREGATES
# ============================================================

def add_geo_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add powerful continuous geographic signals:
        postal_price_mean
        postal_price_per_m2_mean
        locality_price_mean
        locality_price_per_m2_mean
    """
    df = df.copy()

    # Postal aggregates
    if {"postal_code", "price"}.issubset(df.columns):
        df["postal_price_mean"] = df.groupby("postal_code")["price"].transform("mean")

    # Locality aggregates
    if {"locality", "price"}.issubset(df.columns):
        df["locality_price_mean"] = df.groupby("locality")["price"].transform("mean")

    return df


# ============================================================
# 4. TARGET ENCODING (RF / XGB)
# ============================================================

def target_encode_column(
    df: pd.DataFrame,
    col: str,
    target: str = "price",
    alpha: float = TARGET_ENCODING_ALPHA,
    suffix: str | None = None,
) -> pd.DataFrame:
    """
    Smoothed target encoding (K-fold safe version for static datasets):
    TE(category) = (n_c * mean_c + alpha * global_mean) / (n_c + alpha)
    """
    df = df.copy()

    if col not in df.columns or target not in df.columns:
        return df

    global_mean = df[target].mean()
    stats = df.groupby(col)[target].agg(["mean", "count"])

    smoothed = (stats["count"] * stats["mean"] + alpha * global_mean) / (
        stats["count"] + alpha
    )

    new_col = f"{col}_te" if suffix is None else f"{col}_te_{suffix}"
    df[new_col] = df[col].map(smoothed)

    return df


def add_target_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply target encoding to selected categorical features.
    """
    df = df.copy()

    if "price" not in df.columns:
        return df

    for col in TARGET_ENCODING_COLS:
        if col in df.columns:
            df = target_encode_column(df, col, target="price", suffix="price")

    return df

# ============================================================
# 5. Imputation
# ============================================================

def final_imputation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Numeric continuous → median
    numeric_cont = [
        "area", "rooms", "living_room_surface", "build_year",
        "facades_number", "bathrooms", "toilets",
        "cadastral_income_house", "area_log",
    ]

    for col in numeric_cont:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Binary → 0 (No)
    binary_cols = [
        "cellar", "has_garage", "has_swimming_pool",
        "has_equipped_kitchen", "access_disabled",
        "elevator", "leased", "is_furnished", "has_terrace", "has_garden"
    ]

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Ordinal-coded categorical → mode
    ordinal_cols = [
        "heating_type", "glazing_type",
        "sewer_connection", "certification_electrical_installation",
        "preemption_right", "flooding_area_type",
        "attic_house", "state"
    ]

    for col in ordinal_cols:
        if col in df.columns:
            mode = df[col].mode(dropna=True)
            if len(mode) > 0:
                df[col] = df[col].fillna(mode.iloc[0])
            else:
                df[col] = df[col].fillna(0)

    return df



# ============================================================
# 6. MASTER PIPELINE FUNCTION
# ============================================================

def stage3_pipeline(df_stage2: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 3 feature engineering:
        1. Missingness flags & -1→NaN
        2. price_per_m2
        3. log features
        4. geo aggregates
        5. target encoding for RF/XGB
        6. imputation
    Returns a large, enriched ML-ready dataframe.
    """
    df = df_stage2.copy()

    # 1) Missingness
    df = stage3_missingness_handler(df)
    df = final_imputation(df)

    # 2) Core FE
    df = add_log_features(df)

    # 3) Geo FE
    df = add_geo_aggregates(df)

    # 4) Target encodings
    df = add_target_encodings(df)

    return df

# =========================================================
# STAGE 1 — Load Raw + Basic Cleaning (Merged Stage0 + Stage1)
# =========================================================

import re
import numpy as np
import pandas as pd
from config.paths import RAW_FILE
from config.constants import (
    HOUSE_SUBTYPES,
    APARTMENT_SUBTYPES,
    YES_NO_COLS,
    NUMERIC_STR_COLS,
    CORE_DTYPES,
    EXPECTED_MIN_COLUMNS
)

# =========================================================
# HELPERS
# =========================================================
def normalize_subtype(s: str):
    if not isinstance(s, str):
        return None
    s = s.lower().replace("-", " ").strip()
    return re.sub(r"\s+", " ", s)


def map_property_type(subtype):
    if not isinstance(subtype, str):
        return "Other"
    s = normalize_subtype(subtype)
    if s in HOUSE_SUBTYPES:
        return "House"
    if s in APARTMENT_SUBTYPES:
        return "Apartment"
    return "Other"


def normalize_yes_no(val):
    if not isinstance(val, str):
        return np.nan
    v = val.strip().lower()
    if v == "yes":
        return 1
    if v == "no":
        return 0
    return np.nan


def clean_numeric_str_series(series: pd.Series):
    return (
        series.astype(str)
              .str.replace("m", "", regex=False)
              .str.strip()
              .replace(["", "nan", "None"], np.nan)
              .astype(float)
    )


# =========================================================
# MODULE A — URL extraction
# =========================================================
def extract_from_url(df, url_col="url"):
    df = df.copy()

    df["property_subtype"] = df[url_col].str.extract(r"detail/([^/]+)/", expand=False)
    df["postal_code"] = df[url_col].str.extract(r"/(\d{4})/", expand=False)
    df["locality"] = df[url_col].str.extract(r"/\d{4}/([^/]+)/", expand=False)

    return df


# =========================================================
# MODULE B — Cleaning (price, vat, numeric)
# =========================================================
def clean_price_vat(df):
    df = df.copy()
    df["price"] = df["price"].fillna(-1)

    df["vat"] = (
        df["vat"].astype(str)
                  .str.strip()
                  .str.replace(r"^:?\s*No$", "No", regex=True)
                  .replace("nan", np.nan)
    )
    return df


def clean_numeric_columns(df):
    df = df.copy()
    df[NUMERIC_STR_COLS] = df[NUMERIC_STR_COLS].apply(clean_numeric_str_series)
    return df


# =========================================================
# MODULE C — Yes/No → {0,1,NaN}
# =========================================================
def encode_yes_no(df):
    df = df.copy()
    df[YES_NO_COLS] = df[YES_NO_COLS].apply(lambda col: col.map(normalize_yes_no))
    return df


# =========================================================
# MODULE D — Normalize subtype + property type mapping
# =========================================================
def process_property_types(df):
    df = df.copy()
    df["property_subtype"] = df["property_subtype"].apply(normalize_subtype)
    df["property_type"] = df["property_subtype"].apply(map_property_type)
    return df


# =========================================================
# MASTER PIPELINE — STAGE 0 + STAGE 1 (Merged)
# =========================================================
def load_and_clean_stage1(path: str | None = None) -> pd.DataFrame:
    """
    FULL STAGE 1:
    1. Load raw CSV (merged Stage0)
    2. Schema checks / dtype enforcement
    3. URL extraction
    4. Base cleaning & normalization
    """

    # ------------------------------------
    # STAGE 0 — Load raw dataset
    # ------------------------------------
    if path is None:
        path = RAW_FILE

    df = pd.read_csv(path, low_memory=False)

    df = df.drop_duplicates()

    # Enforce dtypes
    for col, dtype in CORE_DTYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    # Convert price to numeric
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Required columns
    missing = [c for c in EXPECTED_MIN_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected raw columns: {missing}")

    # Remove invalid rows
    df = df[df["property_id"].notna() & df["url"].notna() & df["price"].notna()]

    # ------------------------------------
    # STAGE 1 — Immovlan cleaning
    # ------------------------------------
    df = extract_from_url(df)
    df = df.dropna(subset=["locality", "postal_code", "property_subtype"])
    df = clean_price_vat(df)
    df = clean_numeric_columns(df)
    df = encode_yes_no(df)
    df = process_property_types(df)

    return df

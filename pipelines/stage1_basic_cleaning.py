import re
import numpy as np
import pandas as pd

# =========================================================
# 0. CONSTANTS
# =========================================================
HOUSE_SUBTYPES = {
    "residence", "villa", "mixed building", "master house",
    "cottage", "bungalow", "chalet", "mansion"
}

APARTMENT_SUBTYPES = {
    "apartment", "ground floor", "penthouse", "duplex",
    "studio", "loft", "triplex", "student flat", "student housing"
}

YES_NO_COLS = [
    "leased", "running_water", "access_disabled", "preemption_right",
    "has_swimming_pool", "sewer_connection", "attic", "cellar",
    "entry_phone", "solar_panels", "planning_permission_granted",
    "alarm", "heat_pump", "surroundings_protected", "air_conditioning",
    "rain_water_tank", "security_door", "low_energy", "water_softener",
    "opportunity_for_professional"
]

NUMERIC_STR_COLS = ["frontage_width", "terrain_width_roadside"]


# =========================================================
# 1. HELPERS
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
# 2. MODULE A — URL extraction
# =========================================================
def extract_from_url(df, url_col="url"):
    df = df.copy()

    df["property_subtype"] = df[url_col].str.extract(r"detail/([^/]+)/", expand=False)
    df["postal_code"] = df[url_col].str.extract(r"/(\d{4})/", expand=False)
    df["locality"] = df[url_col].str.extract(r"/\d{4}/([^/]+)/", expand=False)

    return df


# =========================================================
# 3. MODULE B — Cleaning (price, vat, numeric)
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
# 4. MODULE C — Yes/No → {0,1,NaN}
# =========================================================
def encode_yes_no(df):
    df = df.copy()
    df[YES_NO_COLS] = df[YES_NO_COLS].apply(lambda col: col.map(normalize_yes_no))
    return df


# =========================================================
# 5. MODULE D — Normalize subtype + property type mapping
# =========================================================
def process_property_types(df):
    df = df.copy()
    df["property_subtype"] = df["property_subtype"].apply(normalize_subtype)
    df["property_type"] = df["property_subtype"].apply(map_property_type)
    return df


# =========================================================
# 6. MASTER PIPELINE — Run everything
# =========================================================
def immovlan_cleaning_pipeline(df_raw):
    """Full ETL pipeline for Immovlan property listings."""
    df = df_raw.drop_duplicates().copy()

    df = extract_from_url(df)
    df = df.dropna(subset=["locality", "postal_code", "property_subtype"])
    df = clean_price_vat(df)
    df = clean_numeric_columns(df)
    df = encode_yes_no(df)
    df = process_property_types(df)

    return df

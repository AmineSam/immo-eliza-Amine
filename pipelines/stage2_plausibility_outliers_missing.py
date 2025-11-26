import pandas as pd
import numpy as np


# =========================================================
# 0. CONSTANTS
# =========================================================

# Columns to drop as non-useful for modeling
COLS_TO_DROP = [
    "parking_places_outdoor", "wash_room", "front_facade_orientation", "diningrooms",
    "parking_places_indoor", "certification_gasoil_tank", "opportunity_for_professional",
    "water_softener", "garden_orientation", "low_energy", "maintenance_cost",
    "terrace_orientation", "security_door", "rain_water_tank", "p_score", "g_score",
    "air_conditioning", "surroundings_protected", "heat_pump", "alarm",
    "terrain_width_roadside", "planning_permission_granted", "frontage_width",
    "solar_panels", "kitchen_type", "garden_surface", "vat", "demarcated_flooding_area",
    "availability",
]

# Binary columns encoded as {1, 0, -1} (missing = -1)
BINARY_COLS = [
    "cellar", "sewer_connection", "has_swimming_pool",
    "preemption_right", "access_disabled", "running_water",
    "is_furnished", "has_garage", "leased", "has_garden",
    "has_terrace",
]

# Numeric columns where missing is encoded as -1
NUMERIC_KEEP_MISSING = [
    "kitchen_surface_house", "living_room_surface",
    "land_surface_house", "apartement_floor_apartment",
    "number_floors_apartment",
]

# Categorical columns where missing is kept as a category, then encoded
CATEGORICAL_KEEP_MISSING = [
    "has_equipped_kitchen",
    "glazing_type",
    "heating_type",
    "state",
]

# Numeric key fields where missing is encoded as -1 for modeling
NUMERIC_KEY_FIELDS = [
    "primary_energy_consumption", "build_year", "bathrooms",
    "elevator", "facades_number", "area", "rooms",
    "terrace_surface_apartment", "co2_house", "attic_house",
    "entry_phone_apartment", "cadastral_income_house", "toilets",
]

# Mapping for state (condition of the property)
STATE_GROUPED_MAPPING = {
    "To renovate": 0,
    "To be renovated": 0,
    "To restore": 0,
    "To demolish": 0,
    "Under construction": 1,
    "Normal": 2,
    "Fully renovated": 3,
    "Excellent": 3,
    "New": 4,
    "Missing": -1,
}

# Mapping for has_equipped_kitchen
KITCHEN_EQUIPPED_MAPPING = {
    "Missing": -1,
    "Not equipped": 0,
    "Partially equipped": 1,
    "Fully equipped": 2,
    "Super equipped": 3,
}

# Mapping for glazing_type
GLAZING_MAPPING = {
    "Missing": -1,
    "Simple glass": 0,
    "Double glass": 1,
    "Triple glass": 2,
}

# Mapping for heating_type
HEATING_MAPPING = {
    "Missing": -1,
    "Not specified": -1,
    "Coal": 0,
    "Wood": 1,
    "Fuel oil": 2,
    "Gas": 3,
    "Hot air": 4,
    "Electricity": 5,
    "Solar energy": 6,
}

# Mapping for flooding_area_type → numeric
FLOODING_MAPPING = {
    "no flooding area": 1,
    "Low risk": 1,
    "(information not available)": -1,
}

# Plausible numeric ranges for automatic outlier detection
PLAUSIBLE_RANGES = {
    # Core information
    "price": (25_000, 5_000_000),                   # realistic price cap 
    "rooms": (0, 15),

    # Surface values
    "area": (15, 1000),                              # 
    "kitchen_surface_house": (0, 200),
    "living_room_surface": (0, 300),
    "land_surface_house": (0, 30_000),              # rare farms can reach 30k m²

    # Bathrooms / toilets
    "bathrooms": (0, 10),
    "toilets": (0, 12),

    # Apartment floor + building floors
    "apartement_floor_apartment": (-1, 50),         # max 50 floors (Belgium < 45)
    "number_floors_apartment": (-1, 50),            # cap to 50 (max in Belgium ≈ 45)

    # Apartment surfaces
    "terrace_surface_apartment": (0, 200),

    # Census / Cadaster
    "cadastral_income_house": (-1, 10_000),         # abnormal values >190k exist → cap at 10k

    # Energy / CO2
    "primary_energy_consumption": (0, 2_000),
    "co2_house": (0, 1_000),                        # after your describe, this is the safe cap

    # Facades (rare but important)
    "facades_number": (-1, 4),                      # 0–4 are realistic
}


# Subtype remapping / filtering
APARTMENT_SUBTYPE_MAP = {
    "ground floor": "apartment",
    "studio": "apartment",
    "student flat": "apartment",
    "loft": "apartment",
}
HOUSE_SUBTYPE_MAP = {
    "residence": "house",
    "master house": "house",
}
# Subtypes to drop entirely
SUBTYPES_TO_DROP = ["mixed building", "mansion"]


# =========================================================
# 1. CORE STEP: Filter to houses & apartments
# =========================================================

def filter_property_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only properties of type House or Apartment.
    Assumes df has a 'property_type' column from Stage 1.
    """
    return df[df["property_type"].isin(["House", "Apartment"])].copy()


# =========================================================
# 2. Drop non-useful columns + rename house/apartment specific ones
# =========================================================

def drop_non_useful_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop hand-picked columns that are not useful for modeling,
    noisy, or sparsely populated.
    """
    return df.drop(columns=COLS_TO_DROP, errors="ignore")


def rename_house_apartment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split shared columns into explicit house/apartment versions
    where needed. This prevents feature leakage between types.
    """
    df = df.copy()

    # House-specific
    df = df.rename(columns={
        "kitchen_surface": "kitchen_surface_house",
        "land_surface": "land_surface_house",
        "garage": "has_garage",
        "co2": "co2_house",
        "cadastral_income": "cadastral_income_house",
        "attic": "attic_house",
    })

    # Apartment-specific
    df = df.rename(columns={
        "entry_phone": "entry_phone_apartment",
        "apartement_floor": "apartement_floor_apartment",
        "terrace_surface": "terrace_surface_apartment",
        "number_floors": "number_floors_apartment",
    })

    return df


# =========================================================
# 3. Missing & encoding logic
# =========================================================

def encode_binary_with_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode binary fields as {1, 0, -1} where -1 represents missing.
    Assumes values are currently {1.0, 0.0, NaN} or similar.
    """
    df = df.copy()
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map({1.0: 1, 0.0: 0})
            df[col] = df[col].fillna(-1).astype(int)
    return df


def fill_numeric_missing_with_minus_one(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Replace NaN by -1 for a given list of numeric columns.
    This keeps missingness explicit while preserving numeric type.
    """
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna(-1)
    return df


def keep_missing_as_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    For selected categorical columns, keep missing as a 'Missing' category.
    This is an intermediate step before numeric encoding.
    """
    df = df.copy()
    for col in CATEGORICAL_KEEP_MISSING:
        if col in df.columns:
            df[col] = df[col].astype("category")
            df[col] = df[col].cat.add_categories("Missing")
            df[col] = df[col].fillna("Missing")
    return df


def encode_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode 'state' as an ordinal variable based on renovation/newness.
    """
    df = df.copy()
    if "state" in df.columns:
        df["state"] = (
            df["state"]
            .fillna("Missing")
            .map(STATE_GROUPED_MAPPING)
        )
    return df


def encode_kitchen_equipped(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode has_equipped_kitchen as an ordinal variable expressing 
    level of equipment.
    """
    df = df.copy()
    if "has_equipped_kitchen" in df.columns:
        df["has_equipped_kitchen"] = (
            df["has_equipped_kitchen"]
            .astype("string")
            .fillna("Missing")
            .map(KITCHEN_EQUIPPED_MAPPING)
        )
    return df


def encode_glazing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode glazing_type as ordinal (simple, double, triple).
    """
    df = df.copy()
    if "glazing_type" in df.columns:
        df["glazing_type"] = (
            df["glazing_type"]
            .astype("string")
            .fillna("Missing")
            .map(GLAZING_MAPPING)
        )
    return df


def encode_heating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode heating_type with an ordinal-like mapping and missing=-1.
    """
    df = df.copy()
    if "heating_type" in df.columns:
        df["heating_type"] = (
            df["heating_type"]
            .astype("string")
            .fillna("Missing")
            .map(HEATING_MAPPING)
        )
    return df


def encode_certification_electrical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode certification_electrical_installation as {1, 0, -1}.
    """
    df = df.copy()
    col = "certification_electrical_installation"
    if col in df.columns:
        df[col] = df[col].map({
            "yes, certificate in accordance": 1,
            "No, certificate does not comply": 0,
        })
        df[col] = df[col].fillna(-1).astype(int)
    return df


def encode_flooding_area(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode flooding_area_type as numeric, treating missing and
    '(information not available)' as -1, and low/no risk as 1.
    """
    df = df.copy()
    col = "flooding_area_type"
    if col in df.columns:
        df[col] = df[col].map(FLOODING_MAPPING)
        df[col] = df[col].fillna(-1).astype(int)
    return df


# =========================================================
# 4. Subtype filtering & remapping
# =========================================================

def normalize_property_subtype(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drop super-rare or structurally different subtypes (mixed building, mansion)
    - Map some granular subtypes to 'apartment' or 'house' for robustness.
    """
    df = df.copy()
    if "property_subtype" not in df.columns:
        return df

    # Drop unwanted subtypes
    df = df[~df["property_subtype"].isin(SUBTYPES_TO_DROP)]

    # Map apartment-like subtypes
    df["property_subtype"] = df["property_subtype"].replace(APARTMENT_SUBTYPE_MAP)
    # Map house-like subtypes
    df["property_subtype"] = df["property_subtype"].replace(HOUSE_SUBTYPE_MAP)

    return df


# =========================================================
# 5. Plausibility rules & outlier detection
# =========================================================

def flag_unusual_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each column in PLAUSIBLE_RANGES, create a *_unusual flag:
    - 1 if value is outside plausible range and not equal to -1 (missing placeholder)
    - 0 otherwise
    """
    df_flagged = df.copy()

    for col, (min_val, max_val) in PLAUSIBLE_RANGES.items():
        if col in df_flagged.columns:
            df_flagged[f"{col}_unusual"] = (
                (df_flagged[col] != -1) &
                df_flagged[col].notna() &
                ((df_flagged[col] < min_val) | (df_flagged[col] > max_val))
            ).astype(int)

    return df_flagged


def split_outliers(df_flagged: pd.DataFrame):
    """
    Using *_unusual columns, split the dataset into:
    - df_cleaned: rows with no unusual flags
    - df_outliers: rows with at least one unusual flag

    Also drops *_unusual columns from both outputs.
    """
    flag_cols = [c for c in df_flagged.columns if c.endswith("_unusual")]

    if not flag_cols:
        # No flags → all rows are clean, no outliers
        return df_flagged, df_flagged.iloc[0:0]

    outlier_mask = df_flagged[flag_cols].sum(axis=1) > 0

    df_outliers = df_flagged[outlier_mask].copy()
    df_cleaned = df_flagged[~outlier_mask].copy()

    # Drop flag columns from both
    df_outliers = df_outliers.drop(columns=flag_cols)
    df_cleaned = df_cleaned.drop(columns=flag_cols)

    return df_cleaned, df_outliers

def cap_values(df: pd.DataFrame, ranges: dict) -> pd.DataFrame:
    """
    Apply hard plausibility capping.
    Values below min or above max become -1 (meaning invalid/impossible).
    """
    df = df.copy()

    for col, (min_val, max_val) in ranges.items():
        if col in df.columns:
            # Apply lower bound
            df.loc[(df[col] != -1) & (df[col] < min_val), col] = -1
            # Apply upper bound
            df.loc[(df[col] != -1) & (df[col] > max_val), col] = -1

    return df


# =========================================================
# 6. MASTER STAGE 2 PIPELINE
# =========================================================

def stage2_pipeline(df_clean: pd.DataFrame):
    """
    Stage 2: plausibility, missing handling, encoding and outlier removal.
    """
    # 1) Filter to House + Apartment
    df = filter_property_types(df_clean)

    # 2) Drop non-useful columns and rename house/apartment-specific features
    df = drop_non_useful_columns(df)
    df = rename_house_apartment_columns(df)

    # 3) Binary encoding & missing handling
    df = encode_binary_with_missing(df)
    df = fill_numeric_missing_with_minus_one(df, NUMERIC_KEEP_MISSING)

    # 4) Categorical → keep 'Missing', then encode
    df = keep_missing_as_category(df)
    df = encode_state(df)
    df = encode_kitchen_equipped(df)
    df = encode_glazing(df)
    df = encode_heating(df)

    # 5) Additional numeric missing handling
    df = fill_numeric_missing_with_minus_one(df, NUMERIC_KEY_FIELDS)

    # 6) Other encodings
    df = encode_certification_electrical(df)
    df = encode_flooding_area(df)

    # 7) Normalize / filter property_subtype
    df = normalize_property_subtype(df)

    # 8) APPLY CAPPING HERE
    df = cap_values(df, PLAUSIBLE_RANGES)
    df = df[df["price"] >= 25000]
    # 9) Flag unusual values & split outliers
    df_flagged = flag_unusual_values(df)
    df_stage2, df_outliers = split_outliers(df_flagged)

    return df_stage2, df_outliers

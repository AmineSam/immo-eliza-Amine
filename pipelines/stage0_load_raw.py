# pipelines/stage0_load_raw.py

import pandas as pd
from config.paths import RAW_FILE
from config.settings import MIN_VALID_PRICE

# Define important dtypes we care about from the start
CORE_DTYPES = {
    "property_id": "string",
    "url": "string",
    # price will be coerced to float with to_numeric
}

EXPECTED_MIN_COLUMNS = [
    "url",
    "property_id",
    "price",
]

def load_raw_dataset(path: str | None = None) -> pd.DataFrame:
    if path is None:
        path = RAW_FILE

    df = pd.read_csv(path, low_memory=False)

    # 1. EARLY duplicate removal — avoids reintroducing invalid rows
    df = df.drop_duplicates()

    # 2. Enforce core dtypes
    for col, dtype in CORE_DTYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    # 3. Price to numeric
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")


    # 5. Minimal schema check
    missing = [c for c in EXPECTED_MIN_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected raw columns: {missing}")

    # 6. Drop rows without identifiers and price
    df = df[df["property_id"].notna() & df["url"].notna() & df["price"].notna()]

    return df


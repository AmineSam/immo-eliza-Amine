# pipelines/stage2_5_geo_enrichment.py

import pandas as pd
import numpy as np

from config.paths import INCOME_PATH, GEO_PATH, POSTAL_PATH, ADDRESS_PATH
from config.constants import (
    PROVINCE_TO_REGION,
    PRICE_TABLE_PROVINCES,
    PRICE_TABLE_REGIONS,
    PRICE_TABLE_BELGIUM
)

# ============================================================
# 0. LOAD EXTERNAL LOOKUPS
# ============================================================

def _load_support_tables():
    # Income (GDP) per municipality
    income_df = pd.read_csv(INCOME_PATH)
    income_df["median_income"] = income_df["median_income"].astype(float)
    income_df["municipality_lower"] = income_df["municipality"].str.lower().str.strip()
    income_df["municipality_upper_lower"] = (
        income_df["municipality_upper"].str.lower().str.strip()
    )

    # Income lookup with both language variants
    dutch_variant = income_df[["municipality_lower", "median_income"]].copy()
    dutch_variant.columns = ["name", "median_income"]

    french_variant = income_df[["municipality_upper_lower", "median_income"]].copy()
    french_variant.columns = ["name", "median_income"]

    income_lookup = pd.concat([dutch_variant, french_variant], ignore_index=True)
    income_lookup = income_lookup.drop_duplicates(subset=["name"])

    # Geo mapping (municipality, arrondissement, province)
    geo_mapping = pd.read_csv(GEO_PATH)
    geo_lookup = geo_mapping[
        [
            "CD_REFNIS",
            "TX_DESCR_NL",
            "TX_DESCR_FR",
            "TX_ADM_DSTR_DESCR_NL",
            "TX_ADM_DSTR_DESCR_FR",
            "TX_PROV_DESCR_NL",
            "TX_PROV_DESCR_FR",
        ]
    ].drop_duplicates()

    geo_lookup["municipality_nl_lower"] = (
        geo_lookup["TX_DESCR_NL"].str.lower().str.strip()
    )
    geo_lookup["municipality_fr_lower"] = (
        geo_lookup["TX_DESCR_FR"].str.lower().str.strip()
    )

    # Postal codes → municipality code
    postal_codes_df = pd.read_csv(POSTAL_PATH, sep=";", encoding="utf-8-sig")
    postal_lookup = postal_codes_df[["Postal Code", "Municipality code"]].copy()
    postal_lookup.columns = ["postal_code", "municipality_code"]
    postal_lookup = postal_lookup.drop_duplicates()
    postal_lookup["postal_code"] = postal_lookup["postal_code"].astype(str)

    # Addresses
    address = pd.read_csv(ADDRESS_PATH)
    address = address.drop(columns=["url"], errors="ignore")

    return income_lookup, geo_lookup, postal_lookup, address


# ============================================================
# 2. BENCHMARK TABLES (Now imported from config.constants)
# ============================================================

def _enrich_with_geo_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applied AFTER all geo/GDP/address merges.
    """
    df = df.copy()

    # STEP 0: price_per_m2
    # Ensure area is numeric and handle potential zeros to avoid inf
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    
    # Avoid division by zero
    df["price_per_m2"] = df["price"] / df["area"].replace(0, np.nan)

    # STEP 1: merge province benchmarks
    df = df.merge(
        PRICE_TABLE_PROVINCES,
        how="left",
        left_on="province_nl",
        right_on="province",
    )
    df = df.rename(columns={
        "apt_avg_m2": "apt_avg_m2_province",
        "house_avg_m2": "house_avg_m2_province",
    })

    # STEP 2: merge region benchmarks
    df = df.merge(PRICE_TABLE_REGIONS, how="left", on="region")
    df = df.rename(columns={
        "apt_avg_m2": "apt_avg_m2_region",
        "house_avg_m2": "house_avg_m2_region",
    })

    # STEP 3: province benchmark selector
    df["province_benchmark_m2"] = np.where(
        df["property_type"] == "Apartment",
        df["apt_avg_m2_province"],
        df["house_avg_m2_province"],
    )

    # STEP 4: region benchmark selector
    df["region_benchmark_m2"] = np.where(
        df["property_type"] == "Apartment",
        df["apt_avg_m2_region"],
        df["house_avg_m2_region"],
    )

    # STEP 5: national benchmark selector
    # Handle case where property_type might be missing or not in table
    def get_national_benchmark(t):
        if not isinstance(t, str):
            return np.nan
        t_lower = t.lower()
        if t_lower in PRICE_TABLE_BELGIUM:
            return PRICE_TABLE_BELGIUM[t_lower]["avg"]
        return np.nan

    df["national_benchmark_m2"] = df["property_type"].apply(get_national_benchmark)

    # STEP 6: engineered features
    df["diff_to_province_avg_m2"]  = df["price_per_m2"] - df["province_benchmark_m2"]
    df["ratio_to_province_avg_m2"] = df["price_per_m2"] / df["province_benchmark_m2"]

    df["diff_to_region_avg_m2"]    = df["price_per_m2"] - df["region_benchmark_m2"]
    df["ratio_to_region_avg_m2"]   = df["price_per_m2"] / df["region_benchmark_m2"]

    df["diff_to_national_avg_m2"]  = df["price_per_m2"] - df["national_benchmark_m2"]
    df["ratio_to_national_avg_m2"] = df["price_per_m2"] / df["national_benchmark_m2"]

    return df


# ============================================================
# 3. MAIN PUBLIC ENTRY POINT: STAGE 2.5
# ============================================================

def stage25_geo_enrichment(df_stage2: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 2.5 — integrate full merging logic into the pipeline:

    - Uses Stage 2 output as 'properties_df'
    - Adds:
        * postal_code → municipality_code
        * municipality → arrondissement/province (NL/FR)
        * median_income (GDP)
        * region
        * address fields
        * benchmarks and engineered features

    Returns an enriched df (subset of Stage 2 rows with income available).
    """
    df = df_stage2.copy()

    # ---- Load external lookups (GDP, geo, postal, address) ----
    income_lookup, geo_lookup, postal_lookup, address = _load_support_tables()

    # ---- Basic filters & helpers ----
    if "locality" in df.columns:
        df["locality_lower"] = df["locality"].str.lower().str.strip()
        df = df.dropna(subset=["locality"])
    
    # Normalize postal_code as string
    df["postal_code"] = df["postal_code"].fillna(0).astype(int).astype(str)
    df.loc[df["postal_code"] == "0", "postal_code"] = None

    # ------------------------------------------------------------------
    # 1) postal_code → municipality_code
    # ------------------------------------------------------------------
    properties_with_muni = df.merge(
        postal_lookup,
        on="postal_code",
        how="left",
    )

    # ------------------------------------------------------------------
    # 2) municipality_code → geo (municipality, arrondissement, province)
    # ------------------------------------------------------------------
    properties_with_geo = properties_with_muni.merge(
        geo_lookup,
        left_on="municipality_code",
        right_on="CD_REFNIS",
        how="left",
    )

    # ------------------------------------------------------------------
    # 3) add median_income, first try Dutch names
    # ------------------------------------------------------------------
    properties_with_geo_gdp = properties_with_geo.merge(
        income_lookup,
        left_on="municipality_nl_lower",
        right_on="name",
        how="left",
    )

    # For unmatched, try French municipality names
    unmatched_mask = properties_with_geo_gdp["median_income"].isna()
    if unmatched_mask.sum() > 0:
        # Filter french_matches to only relevant municipality codes to speed up
        relevant_codes = properties_with_geo_gdp.loc[unmatched_mask, "municipality_code"]
        
        french_matches = properties_with_geo[
            properties_with_geo["municipality_code"].isin(relevant_codes)
        ].merge(
            income_lookup,
            left_on="municipality_fr_lower",
            right_on="name",
            how="inner",
        )

        # Update median_income where found via French name
        # We can use map for faster update. Ensure unique index.
        french_matches = french_matches.drop_duplicates(subset=["municipality_code"])
        income_map = french_matches.set_index("municipality_code")["median_income"]
        properties_with_geo_gdp.loc[unmatched_mask, "median_income"] = \
            properties_with_geo_gdp.loc[unmatched_mask, "municipality_code"].map(income_map)

    # Keep only rows with income data (as per original logic)
    properties_with_geo_gdp = properties_with_geo_gdp[
        properties_with_geo_gdp["median_income"].notna()
    ]

    # Rename geo columns to final names
    properties_with_geo_gdp = properties_with_geo_gdp.rename(columns={
        "TX_DESCR_NL": "municipality_nl",
        "TX_DESCR_FR": "municipality_fr",
        "TX_ADM_DSTR_DESCR_NL": "arrondissement_nl",
        "TX_ADM_DSTR_DESCR_FR": "arrondissement_fr",
        "TX_PROV_DESCR_NL": "province_nl",
        "TX_PROV_DESCR_FR": "province_fr",
    })

    # ------------------------------------------------------------------
    # 4) Region from province (using NL first, then FR)
    # ------------------------------------------------------------------
    properties_with_geo_gdp["province_lang"] = (
        properties_with_geo_gdp["province_nl"]
        .replace("", pd.NA)
        .fillna(properties_with_geo_gdp["province_fr"])
    )

    properties_with_geo_gdp["region"] = properties_with_geo_gdp[
        "province_lang"
    ].map(PROVINCE_TO_REGION)

    # ------------------------------------------------------------------
    # 5) Merge with address table
    # ------------------------------------------------------------------
    properties_with_geo_gdp = properties_with_geo_gdp.merge(
        address,
        on="property_id",
        how="left",
    )

    # ------------------------------------------------------------------
    # 6) Apply benchmark enrichment & engineered features
    # ------------------------------------------------------------------
    df_enriched = _enrich_with_geo_benchmarks(properties_with_geo_gdp)

    # Cleanup intermediate columns
    drop_cols = [
        'locality_lower', 'municipality_code', 'CD_REFNIS',
        'municipality_nl_lower', 'municipality_fr_lower',
        'name', 'province_lang'
        # 'municipality_fr', 'arrondissement_fr', 'province_fr' # Keep these if needed? Original dropped them.
    ]
    df_enriched = df_enriched.drop(columns=drop_cols, errors='ignore')

    return df_enriched

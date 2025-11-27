# pipelines/stage2_5_geo_enrichment.py

import pandas as pd
import numpy as np

# ============================================================
# 0. LOAD EXTERNAL LOOKUPS (same sources as in merging.py)
# ============================================================

# Paths assume pipeline is run from the repo root,
# identical to your original script.
INCOME_PATH = "../data/raw/median_income.csv"
GEO_PATH = "../data/raw/TF_SOC_POP_STRUCT_2025.csv"
POSTAL_PATH = "../data/raw/postal-codes-belgium.csv"
ADDRESS_PATH = "../data/raw/immovlan_addresses.csv"


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
# 1. PROVINCE → REGION MAPPING
# ============================================================

province_to_region = {
    # --- Flanders ---
    "Provincie Antwerpen": "Flanders",
    "Province d’Anvers": "Flanders",
    "Province d'Anvers": "Flanders",

    "Provincie Oost-Vlaanderen": "Flanders",
    "Province de Flandre orientale": "Flanders",

    "Provincie West-Vlaanderen": "Flanders",
    "Province de Flandre occidentale": "Flanders",

    "Provincie Vlaams-Brabant": "Flanders",
    "Province du Brabant flamand": "Flanders",

    "Provincie Limburg": "Flanders",
    "Province du Limbourg": "Flanders",

    # --- Wallonia ---
    "Provincie Henegouwen": "Wallonia",
    "Province du Hainaut": "Wallonia",

    "Provincie Luik": "Wallonia",
    "Province de Liège": "Wallonia",

    "Provincie Namen": "Wallonia",
    "Province de Namur": "Wallonia",

    "Provincie Waals-Brabant": "Wallonia",
    "Province du Brabant wallon": "Wallonia",

    "Provincie Luxemburg": "Wallonia",
    "Province du Luxembourg": "Wallonia",

    # --- Brussels ---
    "Brussels Hoofdstedelijk Gewest": "Brussels",
    "Région de Bruxelles-Capitale": "Brussels",
}


# ============================================================
# 2. BENCHMARK TABLES (unchanged from your script)
# ============================================================

price_table_provinces = pd.DataFrame({
    "province": [
       "Provincie Antwerpen", "Provincie Oost-Vlaanderen",
       "Provincie Henegouwen", "Brussels Hoofdstedelijk Gewest",
       "Provincie Luik", "Provincie West-Vlaanderen", "Provincie Namen",
       "Provincie Waals-Brabant", "Provincie Vlaams-Brabant",
       "Provincie Limburg", "Provincie Luxemburg"
    ],
    "apt_avg_m2":  [2849, 2890, 1847, 3423, 2292, 3760, 2553, 3244, 3260, 2562, 2448],
    "house_avg_m2":[2419, 2257, 1411, 3308, 1708, 2048, 1673, 2342, 2539, 1926, 1620],
})

price_table_regions = pd.DataFrame({
    "region":       ["Flanders", "Wallonia", "Brussels"],
    "apt_avg_m2":   [3133,      2387,       3423],
    "house_avg_m2": [2266,      1642,       3308],
})

price_table_belgium = {
    "apartment": {"avg": 3091, "min": 2188, "max": 4358},
    "house":     {"avg": 2076, "min": 1223, "max": 3296},
}


def _enrich_with_geo_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Your original enrich_with_geo_benchmarks logic,
    applied AFTER all geo/GDP/address merges.
    """
    df = df.copy()

    # STEP 0: price_per_m2
    df["price_per_m2"] = df["price"] / df["area"]

    # STEP 1: merge province benchmarks
    df = df.merge(
        price_table_provinces,
        how="left",
        left_on="province_nl",
        right_on="province",
    )
    df = df.rename(columns={
        "apt_avg_m2": "apt_avg_m2_province",
        "house_avg_m2": "house_avg_m2_province",
    })

    # STEP 2: merge region benchmarks
    df = df.merge(price_table_regions, how="left", on="region")
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
    df["national_benchmark_m2"] = df["property_type"].apply(
        lambda t: price_table_belgium[t.lower()]["avg"]
    )

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
    Stage 2.5 — integrate your full merging.py logic into the pipeline:

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

    # ---- Basic filters & helpers (same as your script) ----
    df["locality_lower"] = df["locality"].str.lower().str.strip()
    df = df.dropna(subset=["price", "locality"])

    # Normalize postal_code as string like in merging.py
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
        french_matches = properties_with_geo[
            properties_with_geo["municipality_code"].isin(
                properties_with_geo_gdp.loc[unmatched_mask, "municipality_code"]
            )
        ].merge(
            income_lookup,
            left_on="municipality_fr_lower",
            right_on="name",
            how="inner",
        )

        for _, row in french_matches.iterrows():
            mask = properties_with_geo_gdp["municipality_code"] == row["municipality_code"]
            properties_with_geo_gdp.loc[mask, "median_income"] = row["median_income"]

    # Keep only rows with income data
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
    ].map(province_to_region)

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

    return df_enriched

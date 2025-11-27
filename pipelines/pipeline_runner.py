# pipelines/pipeline_runner.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config.paths import STAGE1_FILE, STAGE2_FILE
from config.settings import STAGE1_VERSION, STAGE2_VERSION

from pipelines.stage0_load_raw import load_raw_dataset                # :contentReference[oaicite:0]{index=0}
from pipelines.stage1_basic_cleaning import immovlan_cleaning_pipeline  # :contentReference[oaicite:1]{index=1}
from pipelines.stage2_plausibility_outliers_missing import stage2_pipeline  # :contentReference[oaicite:2]{index=2}

# NEW: import the leakage-free Stage 3
from pipelines.stage3_fitted import Stage3Fitter


# =====================================================================
# ORIGINAL RUNNER (kept unchanged for compatibility)
# =====================================================================

def run_full_pipeline(
    raw_path: str | None = None,
    save_intermediate: bool = True
) -> dict[str, pd.DataFrame]:
    """
    Original pipeline runner (Stage 0–2 only).
    Stage 3 is deprecated here because it's not leakage-safe.
    """
    df_raw = load_raw_dataset(raw_path)

    df_stage1 = immovlan_cleaning_pipeline(df_raw)
    df_stage1 = df_stage1.copy()
    df_stage1["__stage1_version"] = STAGE1_VERSION

    df_stage2, df_outliers = stage2_pipeline(df_stage1)
    df_stage2 = df_stage2.copy()
    df_outliers = df_outliers.copy()
    df_stage2["__stage2_version"] = STAGE2_VERSION
    df_outliers["__stage2_version"] = STAGE2_VERSION

    # Save Stage 1 + Stage 2 only
    if save_intermediate:
        df_stage1.to_csv(STAGE1_FILE, index=False)
        df_stage2.to_csv(STAGE2_FILE, index=False)

    return {
        "stage0": df_raw,
        "stage1": df_stage1,
        "stage2": df_stage2,
        "outliers": df_outliers,
    }


# =====================================================================
# Create Price Stratification Bins
# =====================================================================
def _create_price_bins(df, n_bins=10):
    """
    Create stratification bins based on price quantiles.
    Ensures balanced representation of low/mid/high price ranges.
    """
    df = df.copy()
    df["price_bin"] = pd.qcut(df["price"], q=n_bins, duplicates="drop", labels=False)
    return df

# =====================================================================
# NEW — LEAKAGE-SAFE RUNNER WITH STAGE 3 FIT/TRANSFORM
# =====================================================================

def run_full_pipeline_with_split(
    raw_path: str | None = None,
    save_intermediate: bool = True
) -> dict[str, pd.DataFrame | None | Stage3Fitter]:
    """
    Leakage-safe full pipeline:
      Stage0 → Stage1 → Stage2 → (split 70/15/15 stratified by price)
      → Stage3Fitter.fit → Stage3Fitter.transform
    """
    # ---------------------------------------------------------
    # 0) Load raw
    # ---------------------------------------------------------
    df_raw = load_raw_dataset(raw_path)

    # ---------------------------------------------------------
    # 1) Stage 1
    # ---------------------------------------------------------
    df_stage1 = immovlan_cleaning_pipeline(df_raw)
    df_stage1 = df_stage1.copy()
    df_stage1["__stage1_version"] = STAGE1_VERSION

    # ---------------------------------------------------------
    # 2) Stage 2
    # ---------------------------------------------------------
    df_stage2, df_outliers = stage2_pipeline(df_stage1)
    df_stage2 = df_stage2.copy()
    df_outliers = df_outliers.copy()
    df_stage2["__stage2_version"] = STAGE2_VERSION
    df_outliers["__stage2_version"] = STAGE2_VERSION

    # ---------------------------------------------------------
    # 2.5) Stage 2.5
    # ---------------------------------------------------------

    # Load property dataset
    properties_df = df_stage2

    # Load median income (GDP) dataset
    income_df = pd.read_csv('../data/raw/median_income.csv')
    # Load geographic mapping with arrondissement and province info
    geo_mapping = pd.read_csv('../data/raw/TF_SOC_POP_STRUCT_2025.csv')
    # Load postal code to municipality mapping
    postal_codes_df = pd.read_csv('../data/raw/postal-codes-belgium.csv', sep=';', encoding='utf-8-sig')
    # Load property dataset
    address = pd.read_csv('../data/raw/immovlan_addresses.csv')

    # Clean income data
    income_df['median_income'] = income_df['median_income'].astype(float)

    # Standardize municipality names (lowercase for matching)
    income_df['municipality_lower'] = income_df['municipality'].str.lower().str.strip()

    # Standardize locality names (lowercase for matching)
    properties_df['locality_lower'] = properties_df['locality'].str.lower().str.strip()

    # Remove rows with missing price or locality
    properties_df = properties_df.dropna(subset=['price', 'locality'])

    # Step 1: Create postal code to municipality code mapping
    postal_lookup = postal_codes_df[['Postal Code', 'Municipality code']].copy()
    postal_lookup.columns = ['postal_code', 'municipality_code']
    postal_lookup = postal_lookup.drop_duplicates()
    postal_lookup['postal_code'] = postal_lookup['postal_code'].astype(str)

    # Step 2: Create municipality code (CD_REFNIS) to arrondissement/province mapping from geo_mapping
    geo_lookup = geo_mapping[['CD_REFNIS', 'TX_DESCR_NL', 'TX_DESCR_FR', 
                            'TX_ADM_DSTR_DESCR_NL', 'TX_ADM_DSTR_DESCR_FR', 
                            'TX_PROV_DESCR_NL', 'TX_PROV_DESCR_FR']].drop_duplicates()

    # Standardize municipality names for matching with income data
    geo_lookup['municipality_nl_lower'] = geo_lookup['TX_DESCR_NL'].str.lower().str.strip()
    geo_lookup['municipality_fr_lower'] = geo_lookup['TX_DESCR_FR'].str.lower().str.strip()

    # NEW MERGING STRATEGY: Use postal codes to get municipality codes, then merge with geographic data

    # Step 1: Prepare properties data with postal codes
    # Convert postal codes to integers (they're stored as floats like 2660.0)
    properties_df['postal_code'] = properties_df['postal_code'].fillna(0).astype(int).astype(str)
    properties_df.loc[properties_df['postal_code'] == '0', 'postal_code'] = None

    # Step 2: Merge properties with postal code lookup to get municipality codes
    properties_with_muni = properties_df.merge(
        postal_lookup,
        on='postal_code',
        how='left'
    )

    # Step 3: Merge with geo_lookup to get arrondissement and province
    properties_with_geo = properties_with_muni.merge(
        geo_lookup,
        left_on='municipality_code',
        right_on='CD_REFNIS',
        how='left'
    )
    # Step 5: Prepare income lookup with both Dutch and French names
    income_df['municipality_lower'] = income_df['municipality'].str.lower().str.strip()
    income_df['municipality_upper_lower'] = income_df['municipality_upper'].str.lower().str.strip()

    # Create comprehensive income lookup
    income_lookup = pd.DataFrame()
    dutch_variant = income_df[['municipality_lower', 'median_income']].copy()
    dutch_variant.columns = ['name', 'median_income']
    french_variant = income_df[['municipality_upper_lower', 'median_income']].copy()
    french_variant.columns = ['name', 'median_income']
    income_lookup = pd.concat([dutch_variant, french_variant], ignore_index=True)
    income_lookup = income_lookup.drop_duplicates(subset=['name'])

    # Step 6: Merge municipality prices with income data
    # Try Dutch names first
    properties_with_geo_gdp = properties_with_geo.merge(
        income_lookup,
        left_on='municipality_nl_lower',
        right_on='name',
        how='left'
    )

    # Try French names for unmatched
    unmatched_mask = properties_with_geo_gdp['median_income'].isna()
    if unmatched_mask.sum() > 0:
        french_matches = properties_with_geo[properties_with_geo['municipality_code'].isin(
            properties_with_geo_gdp.loc[unmatched_mask, 'municipality_code']
        )].merge(
            income_lookup,
            left_on='municipality_fr_lower',
            right_on='name',
            how='inner'
        )
        
        # Update the unmatched records with French name matches
        for idx, row in french_matches.iterrows():
            mask = properties_with_geo_gdp['municipality_code'] == row['municipality_code']
            properties_with_geo_gdp.loc[mask, 'median_income'] = row['median_income']

    # Remove unmatched municipalities (keep only those with income data)
    properties_with_geo_gdp = properties_with_geo_gdp[properties_with_geo_gdp['median_income'].notna()]

    properties_with_geo_gdp = properties_with_geo_gdp.rename(columns={
        'TX_DESCR_NL': 'municipality_nl',
        'TX_DESCR_FR': 'municipality_fr',
        'TX_ADM_DSTR_DESCR_NL': 'arrondissement_nl',
        'TX_ADM_DSTR_DESCR_FR': 'arrondissement_fr',
        'TX_PROV_DESCR_NL': 'province_nl',
        'TX_PROV_DESCR_FR': 'province_fr'
    })


    #Regions
    province_to_region = {
        # --- Flanders ---
        'Provincie Antwerpen': 'Flanders',
        'Province d’Anvers': 'Flanders',
        'Province d\'Anvers': 'Flanders',  # safe ASCII fallback

        'Provincie Oost-Vlaanderen': 'Flanders',
        'Province de Flandre orientale': 'Flanders',

        'Provincie West-Vlaanderen': 'Flanders',
        'Province de Flandre occidentale': 'Flanders',

        'Provincie Vlaams-Brabant': 'Flanders',
        'Province du Brabant flamand': 'Flanders',

        'Provincie Limburg': 'Flanders',
        'Province du Limbourg': 'Flanders',

        # --- Wallonia ---
        'Provincie Henegouwen': 'Wallonia',
        'Province du Hainaut': 'Wallonia',

        'Provincie Luik': 'Wallonia',
        'Province de Liège': 'Wallonia',

        'Provincie Namen': 'Wallonia',
        'Province de Namur': 'Wallonia',

        'Provincie Waals-Brabant': 'Wallonia',
        'Province du Brabant wallon': 'Wallonia',

        'Provincie Luxemburg': 'Wallonia',
        'Province du Luxembourg': 'Wallonia',

        # --- Brussels ---
        'Brussels Hoofdstedelijk Gewest': 'Brussels',
        'Région de Bruxelles-Capitale': 'Brussels'
    }

    properties_with_geo_gdp['province_lang'] = (
        properties_with_geo_gdp['province_nl']
        .replace('', pd.NA)
        .fillna(properties_with_geo_gdp['province_fr'])
    )

    properties_with_geo_gdp['region'] = (
        properties_with_geo_gdp['province_lang'].map(province_to_region)
    )

    # Step : Merge with address
    address.drop(columns=['url'], inplace=True)

    properties_with_geo_gdp = properties_with_geo_gdp.merge(
        address,
        on='property_id',
        how='left'
    )


    # ============================================================
    # 1. Your exact benchmark tables
    # ============================================================

    price_table_provinces = pd.DataFrame({
        "province": [
        'Provincie Antwerpen', 'Provincie Oost-Vlaanderen',
        'Provincie Henegouwen', 'Brussels Hoofdstedelijk Gewest',
        'Provincie Luik', 'Provincie West-Vlaanderen', 'Provincie Namen',
        'Provincie Waals-Brabant', 'Provincie Vlaams-Brabant',
        'Provincie Limburg', 'Provincie Luxemburg'
        ],
        "apt_avg_m2":  [2849, 2890, 1847, 3423, 2292, 3760, 2553, 3244, 3260, 2562, 2448],
        "house_avg_m2":[2419, 2257, 1411, 3308, 1708, 2048, 1673, 2342, 2539, 1926, 1620]
    })

    price_table_regions = pd.DataFrame({
        "region":       ['Flanders', 'Wallonia', 'Brussels'],
        "apt_avg_m2":   [3133,      2387,       3423],
        "house_avg_m2": [2266,      1642,       3308]
    })

    # National benchmark values
    price_table_belgium = {
        "apartment": {"avg": 3091, "min": 2188, "max": 4358},
        "house":     {"avg": 2076, "min": 1223, "max": 3296}
    }

    # ============================================================
    # 2. Enrichment Function (creates price_per_m2)
    # ============================================================

    def enrich_with_geo_benchmarks(df):
        df = df.copy()

        # --------------------------------------------------------
        # STEP 0: Compute price_per_m2 (needed for engineered features)
        # --------------------------------------------------------
        df["price_per_m2"] = df["price"] / df["area"]

        # --------------------------------------------------------
        # STEP 1: Merge provinces
        # --------------------------------------------------------
        df = df.merge(
            price_table_provinces,
            how="left",
            left_on="province_nl",
            right_on="province"
        )

        df = df.rename(columns={
            "apt_avg_m2": "apt_avg_m2_province",
            "house_avg_m2": "house_avg_m2_province"
        })


        # --------------------------------------------------------
        # STEP 2: Merge regions
        # --------------------------------------------------------
        df = df.merge(
            price_table_regions,
            how="left",
            on="region"
        )

        df = df.rename(columns={
            "apt_avg_m2": "apt_avg_m2_region",
            "house_avg_m2": "house_avg_m2_region"
        })


        # --------------------------------------------------------
        # STEP 3: Province benchmark (Apartment vs House)
        # --------------------------------------------------------
        df["province_benchmark_m2"] = np.where(
            df["property_type"] == "Apartment",
            df["apt_avg_m2_province"],
            df["house_avg_m2_province"]
        )

        # --------------------------------------------------------
        # STEP 4: Region benchmark
        # --------------------------------------------------------
        df["region_benchmark_m2"] = np.where(
            df["property_type"] == "Apartment",
            df["apt_avg_m2_region"],
            df["house_avg_m2_region"]
        )

        # --------------------------------------------------------
        # STEP 5: National benchmark
        # --------------------------------------------------------
        df["national_benchmark_m2"] = df["property_type"].apply(
            lambda t: price_table_belgium[t.lower()]["avg"]
        )

        # --------------------------------------------------------
        # STEP 6: Engineered ML features
        # --------------------------------------------------------

        # Province-based
        df["diff_to_province_avg_m2"]  = df["price_per_m2"] - df["province_benchmark_m2"]
        df["ratio_to_province_avg_m2"] = df["price_per_m2"] / df["province_benchmark_m2"]

        # Region-based
        df["diff_to_region_avg_m2"]    = df["price_per_m2"] - df["region_benchmark_m2"]
        df["ratio_to_region_avg_m2"]   = df["price_per_m2"] / df["region_benchmark_m2"]

        # National-based
        df["diff_to_national_avg_m2"]  = df["price_per_m2"] - df["national_benchmark_m2"]
        df["ratio_to_national_avg_m2"] = df["price_per_m2"] / df["national_benchmark_m2"]

        return df


    # ============================================================
    # 3. Usage
    # ============================================================


    #drop unecessary columns from previous merges
    properties_with_geo_gdp.drop(columns=['locality_lower', 'municipality_code', 'CD_REFNIS',
       'municipality_fr', 'arrondissement_fr',
       'province_fr', 'municipality_nl_lower',
       'municipality_fr_lower', 'name', 'province_lang'], inplace=True)
    df_stage2 = enrich_with_geo_benchmarks(properties_with_geo_gdp)


    if save_intermediate:
        df_stage1.to_csv(STAGE1_FILE, index=False)
        df_stage2.to_csv(STAGE2_FILE, index=False)

    # ---------------------------------------------------------
    # 3) STRATIFIED SPLIT (70 train / 15 val / 15 test)
    # ---------------------------------------------------------

    # Create price bins for stratification
    df_stage2 = _create_price_bins(df_stage2, n_bins=10)

    # First split: train (70%) and temp (30%)
    df_train_stage2, df_temp_stage2 = train_test_split(
        df_stage2,
        test_size=0.30,
        random_state=42,
        stratify=df_stage2["price_bin"]
    )

    # Second split: val (15%) and test (15%)
    df_val_stage2, df_test_stage2 = train_test_split(
        df_temp_stage2,
        test_size=0.50,
        random_state=42,
        stratify=df_temp_stage2["price_bin"]
    )

    # Remove helper column
    df_train_stage2 = df_train_stage2.drop(columns=['url', 'property_id', '__stage1_version', '__stage2_version',
       'address', 'price_per_m2', 'province', 'price_bin', 'diff_to_province_avg_m2', 'ratio_to_province_avg_m2',
       'diff_to_region_avg_m2', 'ratio_to_region_avg_m2',
       'diff_to_national_avg_m2', 'ratio_to_national_avg_m2'] )
    df_val_stage2   = df_val_stage2.drop(columns=['url', 'property_id', '__stage1_version', '__stage2_version',
       'address', 'price_per_m2', 'province', 'price_bin', 'diff_to_province_avg_m2', 'ratio_to_province_avg_m2',
       'diff_to_region_avg_m2', 'ratio_to_region_avg_m2',
       'diff_to_national_avg_m2', 'ratio_to_national_avg_m2'] )
    df_test_stage2  = df_test_stage2.drop(columns=['url', 'property_id', '__stage1_version', '__stage2_version',
       'address', 'price_per_m2', 'province', 'price_bin', 'diff_to_province_avg_m2', 'ratio_to_province_avg_m2',
       'diff_to_region_avg_m2', 'ratio_to_region_avg_m2',
       'diff_to_national_avg_m2', 'ratio_to_national_avg_m2'] )
    # ---------------------------------------------------------
    # 4) Stage 3
    # ---------------------------------------------------------
    
    s3 = Stage3Fitter()

    # Fit on train only
    s3.fit(df_train_stage2)

    # Transform all splits
    df_train_stage3 = s3.transform(df_train_stage2)
    df_val_stage3   = s3.transform(df_val_stage2)
    df_test_stage3  = s3.transform(df_test_stage2) 

    # ---------------------------------------------------------
    # 5) Return everything
    # ---------------------------------------------------------
    return {
        "raw": df_raw,
        "stage1": df_stage1,
        "stage2": df_stage2,
        "outliers": df_outliers,
        "train": df_train_stage3,
        "val": df_val_stage3, # will be None
        "test": df_test_stage3,
        "stage3_fitter": s3,
    }

# pipelines/pipeline_runner.py

import pandas as pd
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
    df_train_stage2 = df_train_stage2.drop(columns=["price_bin"])
    df_val_stage2   = df_val_stage2.drop(columns=["price_bin"])
    df_test_stage2  = df_test_stage2.drop(columns=["price_bin"])

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

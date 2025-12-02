import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config.paths import STAGE1_FILE, STAGE2_FILE
from config.settings import STAGE1_VERSION, STAGE2_VERSION

from pipelines.stage1_basic_cleaning import load_and_clean_stage1
from pipelines.stage2_plausibility_outliers_missing import stage2_pipeline
from pipelines.stage2_5_geo_enrichment import stage25_geo_enrichment
from pipelines.stage3_fitted import Stage3Fitter

# =====================================================================
# LEAKAGE-SAFE RUNNER WITH STAGE 3 FIT/TRANSFORM
# =====================================================================

def run_full_pipeline_with_split(
    raw_path: str | None = None,
    save_intermediate: bool = True
) -> dict[str, pd.DataFrame | None | Stage3Fitter]:
    """
    Leakage-safe full pipeline:
      Stage0 → Stage1 → Stage2 → Stage2.5 (Geo) → (split 70/15/15 stratified by price)
      → Stage3Fitter.fit → Stage3Fitter.transform
    """
    # ---------------------------------------------------------
    # 1) Stage 1
    # ---------------------------------------------------------
    print("Running Stage 1...")
    df_stage1 = load_and_clean_stage1(raw_path)
    df_stage1 = df_stage1.copy()
    df_stage1["__stage1_version"] = STAGE1_VERSION

    # ---------------------------------------------------------
    # 2) Stage 2
    # ---------------------------------------------------------
    print("Running Stage 2...")
    df_stage2 = stage2_pipeline(df_stage1)
    df_stage2 = df_stage2.copy()
    df_stage2["__stage2_version"] = STAGE2_VERSION

    # ---------------------------------------------------------
    # 2.5) Stage 2.5 — GEO + GDP ENRICHMENT (NO PRICE USED)
    # ---------------------------------------------------------
    print("Running Stage 2.5 (Geo Enrichment)...")
    df_stage2_enriched = stage25_geo_enrichment(df_stage2)

    # Save intermediates if needed
    if save_intermediate:
        print("Saving intermediate files...")
        df_stage1.to_csv(STAGE1_FILE, index=False)
        # Save the enriched stage 2 as the final stage 2 output
        df_stage2_enriched.to_csv(STAGE2_FILE, index=False)

    # ---------------------------------------------------------
    # 3) SPLIT (70 / 15 / 15) — now on leakage-safe Stage2
    # ---------------------------------------------------------
    print("Splitting data...")
    # Drop technical columns before split
    df_to_split = df_stage2_enriched.drop(
        columns=[
            'url', 'property_id', '__stage1_version', '__stage2_version',
            'address'
        ],
        errors='ignore'
    )

    df_train_stage2, df_temp_stage2 = train_test_split(
        df_to_split,
        test_size=0.30,
        random_state=42,
    )

    df_val_stage2, df_test_stage2 = train_test_split(
        df_temp_stage2,
        test_size=0.50,
        random_state=42,
    )

    # ---------------------------------------------------------
    # 4) Stage 3 — fit on train, transform all
    # ---------------------------------------------------------
    print("Running Stage 3 (Fitted)...")
    s3 = Stage3Fitter()
    s3.fit(df_train_stage2)

    df_train_stage3 = s3.transform(df_train_stage2)
    df_val_stage3   = s3.transform(df_val_stage2)
    df_test_stage3  = s3.transform(df_test_stage2)

    print("Pipeline complete.")
    return {
        "stage1": df_stage1,
        "stage2": df_stage2_enriched,
        "train": df_train_stage3,
        "val": df_val_stage3,
        "test": df_test_stage3,
        "stage3_fitter": s3,
        "raw": None # Included for compatibility if needed, though raw isn't returned by stage1 loader directly
    }

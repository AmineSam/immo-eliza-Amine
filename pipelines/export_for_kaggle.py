"""
Export script for Kaggle
Generates a pre-processed CSV file ready for upload to Kaggle.
Runs Stages 1, 2, and 2.5 (geo enrichment) only.
"""

import pandas as pd
from pathlib import Path

from pipelines.stage1_basic_cleaning import load_and_clean_stage1
from pipelines.stage2_plausibility_outliers_missing import stage2_pipeline
from pipelines.stage2_5_geo_enrichment import stage25_geo_enrichment

def export_for_kaggle(output_path: str = "../data/pre_processed/pre_processed_data_for_kaggle.csv"):
    """
    Run the pipeline up to Stage 2.5 and save the result.
    This CSV can be uploaded to Kaggle for model training.
    """
    print("=" * 60)
    print("Exporting pre-processed data for Kaggle")
    print("=" * 60)
    
    # Stage 1: Basic cleaning
    print("\n[1/3] Running Stage 1 (Basic Cleaning)...")
    df_stage1 = load_and_clean_stage1()
    print(f"   ✓ Stage 1 complete: {df_stage1.shape[0]:,} rows, {df_stage1.shape[1]} columns")
    
    # Stage 2: Plausibility, outliers, missing
    print("\n[2/3] Running Stage 2 (Plausibility & Outliers)...")
    df_stage2 = stage2_pipeline(df_stage1)
    print(f"   ✓ Stage 2 complete: {df_stage2.shape[0]:,} rows, {df_stage2.shape[1]} columns")
    
    # Stage 2.5: Geo enrichment
    print("\n[3/3] Running Stage 2.5 (Geo Enrichment)...")
    df_enriched = stage25_geo_enrichment(df_stage2)
    print(f"   ✓ Stage 2.5 complete: {df_enriched.shape[0]:,} rows, {df_enriched.shape[1]} columns")
    
    # Save to CSV
    print(f"\n[SAVE] Writing to {output_path}...")
    df_enriched.to_csv(output_path, index=False)
    
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"   ✓ Saved successfully ({file_size_mb:.2f} MB)")
    
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    
    return df_enriched

if __name__ == "__main__":
    export_for_kaggle()

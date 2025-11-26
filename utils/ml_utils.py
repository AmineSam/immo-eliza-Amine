import pandas as pd
import numpy as np

from pipelines.stage3_feature_engineering import stage3_pipeline


def prepare_stage3_dataset(stage2_path: str) -> pd.DataFrame:
    """Load Stage 2, run Stage 3, return enriched df."""
    df_stage2 = pd.read_csv('../data/clean/clean_dataset_stage3_v4.csv')
    df_stage3 = stage3_pipeline(df_stage2)
    return df_stage3


def prepare_X_y(df: pd.DataFrame, model_type: str = "linear"):
    """
    Build X, y for a given model type.
    
    model_type:
      - "linear": Linear Regression (numeric only, no TE)
      - "rf": RandomForest (numeric + TE)
      - "xgb": XGBoost (numeric + TE)
    """
    df = df.copy()

    # 1) Drop technical / useless columns
    drop_cols = [
        "property_id", "url"
    ]
    # Drop any __stage versioning columns
    drop_cols += [c for c in df.columns if c.startswith("__stage")]
    df = df.drop(columns=drop_cols, errors="ignore")

    # 2) Separate target
    if "price" not in df.columns:
        raise ValueError("Column 'price' not found in dataframe.")
    y = df["price"]
    X = df.drop(columns=["price"])

    # 3) Remove leakage features
    #    Anything directly derived from price
    leakage_cols = []
    leakage_cols += [c for c in X.columns if c.startswith("price_per_m2")]
    leakage_cols += ["price_log"]
    X = X.drop(columns=leakage_cols, errors="ignore")

    # 4) Model-type specific filtering
    if model_type == "linear":
        # For LR: numeric only, no TE, no raw high-cardinality categoricals
        X = X.select_dtypes(include=[np.number])
        # Drop target-encoding columns if you want LR to be “pure”
        X = X.drop(
            columns=[c for c in X.columns if c.endswith("_te_price")],
            errors="ignore",
        )

    elif model_type in ("rf", "xgb"):
        # For trees: numeric only, TE is useful
        X = X.select_dtypes(include=[np.number])
        # We keep *_te_price and geo aggregates.
        # If you want, you can also drop log columns to avoid redundancy:
        X = X.drop(columns=[c for c in X.columns if c.endswith("_log")], errors="ignore")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return X, y

# pipelines/stage3_fitted.py

import pandas as pd
import numpy as np

from pipelines.stage3_feature_engineering import (
    MISSINGNESS_NUMERIC_COLS,
    LOG_FEATURES,
    TARGET_ENCODING_COLS,
    TARGET_ENCODING_ALPHA,
    add_missingness_flags,
    convert_minus1_to_nan,
    add_price_per_m2,
    add_log_features,
)


class Stage3Fitter:
    """
    Leakage-safe Stage 3 implementation.
    Structure is IDENTICAL to your stage3_feature_engineering logic,
    but separated into fit/transform while respecting your constants,
    helper functions, feature names, and column lists.
    """

    # ---------------------------------------------------------
    # INTERNAL STATE FITTED FROM TRAIN ONLY
    # ---------------------------------------------------------
    postal_price_mean_: dict = None
    postal_ppm2_mean_: dict = None
    locality_price_mean_: dict = None
    locality_ppm2_mean_: dict = None

    te_maps_: dict = None
    global_mean_: float = None

    numeric_medians_: dict = None
    ordinal_modes_: dict = None

    # ---------------------------------------------------------
    # IMPUTATION COLUMN GROUPS (copied 1:1 from final_imputation)
    # ---------------------------------------------------------
    @staticmethod
    def _numeric_cont_columns():
        return [
            "area", "rooms", "living_room_surface", "build_year",
            "facades_number", "bathrooms", "toilets",
            "cadastral_income_house", "area_log",
        ]

    @staticmethod
    def _binary_columns():
        return [
            "cellar", "has_garage", "has_swimming_pool",
            "has_equipped_kitchen", "access_disabled",
            "elevator", "leased", "is_furnished",
            "has_terrace", "has_garden",
        ]

    @staticmethod
    def _ordinal_columns():
        return [
            "heating_type", "glazing_type",
            "sewer_connection", "certification_electrical_installation",
            "preemption_right", "flooding_area_type",
            "attic_house", "state",
        ]

    # ---------------------------------------------------------
    # FIT IMPUTERS ON TRAIN
    # ---------------------------------------------------------
    def _fit_imputers(self, df: pd.DataFrame):
        self.numeric_medians_ = {}
        self.ordinal_modes_ = {}

        for col in self._numeric_cont_columns():
            if col in df.columns:
                self.numeric_medians_[col] = df[col].median()

        for col in self._ordinal_columns():
            if col in df.columns:
                mode = df[col].mode(dropna=True)
                self.ordinal_modes_[col] = mode.iloc[0] if len(mode) > 0 else 0

    # ---------------------------------------------------------
    # APPLY IMPUTERS USING TRAIN STATS
    # ---------------------------------------------------------
    def _apply_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # numeric → median
        for col, med in self.numeric_medians_.items():
            if col in df.columns:
                df[col] = df[col].fillna(med)

        # binary → 0
        for col in self._binary_columns():
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # ordinal → mode
        for col, mode in self.ordinal_modes_.items():
            if col in df.columns:
                df[col] = df[col].fillna(mode)

        return df

    # ---------------------------------------------------------
    # FIT GEO
    # ---------------------------------------------------------
    def _fit_geo(self, df: pd.DataFrame):
        self.postal_price_mean_ = df.groupby("postal_code")["price"].mean().to_dict()
        self.postal_ppm2_mean_ = df.groupby("postal_code")["price_per_m2"].mean().to_dict()

        self.locality_price_mean_ = df.groupby("locality")["price"].mean().to_dict()
        self.locality_ppm2_mean_ = df.groupby("locality")["price_per_m2"].mean().to_dict()

    # ---------------------------------------------------------
    # APPLY GEO (train-fitted)
    # ---------------------------------------------------------
    def _apply_geo(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["postal_price_mean"] = df["postal_code"].map(self.postal_price_mean_)
        df["postal_price_per_m2_mean"] = df["postal_code"].map(self.postal_ppm2_mean_)

        df["locality_price_mean"] = df["locality"].map(self.locality_price_mean_)
        df["locality_price_per_m2_mean"] = df["locality"].map(self.locality_ppm2_mean_)

        return df

    # ---------------------------------------------------------
    # TARGET ENCODING — FIT
    # ---------------------------------------------------------
    def _fit_target_encoding(self, df: pd.DataFrame):
        self.te_maps_ = {}
        self.global_mean_ = df["price"].mean()

        for col in TARGET_ENCODING_COLS:
            stats = df.groupby(col)["price"].agg(["mean", "count"])
            smoothed = (
                stats["count"] * stats["mean"]
                + TARGET_ENCODING_ALPHA * self.global_mean_
            ) / (stats["count"] + TARGET_ENCODING_ALPHA)

            self.te_maps_[col] = smoothed.to_dict()

    # ---------------------------------------------------------
    # TARGET ENCODING — APPLY
    # ---------------------------------------------------------
    def _apply_target_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col, mapping in self.te_maps_.items():
            out_col = f"{col}_te_price"
            df[out_col] = df[col].map(mapping).fillna(self.global_mean_)

        return df

    # ---------------------------------------------------------
    # FIT (TRAIN ONLY)
    # ---------------------------------------------------------
    def fit(self, df_train_stage2: pd.DataFrame):
        df = df_train_stage2.copy()

        # 1) Missingness
        df = add_missingness_flags(df)
        df = convert_minus1_to_nan(df)

        # 2) EARLY IMPUTATION (stabilizes FE)
        #    fits imputers on the cleaned training set
        self._fit_imputers(df)
        df = self._apply_imputation(df)

        # 3) Core FE
        df = add_price_per_m2(df)
        df = add_log_features(df)

        # 4) Fit GEO + TE (train only)
        self._fit_geo(df)
        self._fit_target_encoding(df)

        # 5) Fit FINAL imputers using FE output
        self._fit_imputers(df)

    # ---------------------------------------------------------
    # TRANSFORM (TRAIN/VAL/TEST)
    # ---------------------------------------------------------
    def transform(self, df_stage2: pd.DataFrame) -> pd.DataFrame:
        df = df_stage2.copy()

        # 1) Missingness
        df = add_missingness_flags(df)
        df = convert_minus1_to_nan(df)

        # 2) EARLY IMPUTATION
        df = self._apply_imputation(df)

        # 3) Core FE
        df = add_price_per_m2(df)
        df = add_log_features(df)

        # 4) GEO + TE (creates NaNs)
        df = self._apply_geo(df)
        df = self._apply_target_encoding(df)

        # 5) FINAL IMPUTATION (fix NaNs from FE/TE/GEO)
        df = self._apply_imputation(df)

        return df

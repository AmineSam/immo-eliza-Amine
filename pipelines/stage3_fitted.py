# pipelines/stage3_fitted.py

import pandas as pd
import numpy as np

    # ---------------------------------------------------------
    # INTERNAL STATE FITTED FROM TRAIN ONLY
    # ---------------------------------------------------------
    te_maps_: dict = None
    global_mean_: float = None

    numeric_medians_: dict = None
    ordinal_modes_: dict = None

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
            'cadastral_income_house', 'median_income',
            'apt_avg_m2_province', 'house_avg_m2_province',
            'apt_avg_m2_region', 'house_avg_m2_region',
            'province_benchmark_m2', 'region_benchmark_m2', 'national_benchmark_m2',
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
            "attic_house", "state",  'region', 'province',
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
    # ---------------------------------------------------------
    # FIT GEO
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # FIT GEO (REMOVED LEAKAGE)
    # ---------------------------------------------------------
    # def _fit_geo(self, df: pd.DataFrame):
    #     pass 

    # ---------------------------------------------------------
    # APPLY GEO (REMOVED LEAKAGE)
    # ---------------------------------------------------------
    # def _apply_geo(self, df: pd.DataFrame) -> pd.DataFrame:
    #     return df.copy()

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
        df = add_log_features(df)

        # 4) Fit GEO + TE (train only)
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
        df = add_log_features(df)

        # 4) GEO + TE (creates NaNs)
        df = self._apply_target_encoding(df)

        # 5) FINAL IMPUTATION (fix NaNs from FE/TE/GEO)
        df = self._apply_imputation(df)

        return df

import pandas as pd
import joblib
from utils.stage3_utils import transform_stage3

REDUCED_FEATURES = [
    "area",
    "postal_code_te_price",
    "locality_te_price",
    "bathrooms",
    "rooms",
    "primary_energy_consumption",
    "state",
    "province_benchmark_m2",
    "postal_code",
    "region_benchmark_m2",
    "property_subtype_te_price",
    "apt_avg_m2_region",
    "toilets",
    "property_type_te_price",
    "median_income",
    "build_year",
    "house_avg_m2_province",
    "has_garage",
    "apt_avg_m2_province",
    "has_garden",
    "has_terrace",
    "facades_number",
    "has_swimming_pool",
    "house_avg_m2_region",
    "has_equipped_kitchen",
]

# Load models + pipelines once globally
model_house = joblib.load("../models/model_xgb_house.pkl")
stage3_house = joblib.load("../models/stage3_pipeline_house.pkl")

model_apartment = joblib.load("../models/model_xgb_apartment.pkl")
stage3_apartment = joblib.load("../models/stage3_pipeline_apartment.pkl")


def predict_price(property_dict):
    df = pd.DataFrame([property_dict])

    if df.loc[0, "property_type"] == "House":
        df_s3 = transform_stage3(df, stage3_house)
        model = model_house
    else:
        df_s3 = transform_stage3(df, stage3_apartment)
        model = model_apartment

    X = df_s3[[f for f in REDUCED_FEATURES if f in df_s3.columns]]
    return float(model.predict(X)[0])

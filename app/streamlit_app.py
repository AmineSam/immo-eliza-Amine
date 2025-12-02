import streamlit as st
import pandas as pd
import sys, os

# Add project root to path
sys.path.append(os.path.abspath(".."))

from utils.predictor import predict_price   # <--- Unified predictor


# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Immo-Eliza Price Predictor",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 Immo-Eliza Real Estate Price Predictor")
st.write("Predict the market price of **Houses** and **Apartments** in Belgium using your trained ML models.")


# ---------------------------------------------------------
# PROPERTY TYPE SELECTION
# ---------------------------------------------------------
property_type = st.selectbox(
    "Select Property Type",
    ["Apartment", "House"]
)

st.markdown("---")

# ---------------------------------------------------------
# INPUT FORM
# ---------------------------------------------------------
st.subheader(f"{property_type} Characteristics")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (m²)", min_value=20, max_value=1000, value=100)
    rooms = st.number_input("Number of Rooms", min_value=1, max_value=20, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=1)
    toilets = st.number_input("Toilets", min_value=0, max_value=10, value=1)
    build_year = st.number_input("Build Year", min_value=1800, max_value=2025, value=2000)
    primary_energy_consumption = st.number_input("Primary Energy Consumption", min_value=0, max_value=1000, value=250)

with col2:
    state = st.selectbox("State (Condition)", [1, 2, 3, 4, 5], index=2)
    postal_code = st.number_input("Postal Code", min_value=1000, max_value=9999, value=1050)
    facades_number = st.number_input("Number of Facades", min_value=1, max_value=8, value=2)
    has_garage = st.selectbox("Garage?", [0, 1])
    has_garden = st.selectbox("Garden?", [0, 1])
    has_terrace = st.selectbox("Terrace?", [0, 1])
    has_swimming_pool = st.selectbox("Swimming Pool?", [0, 1])
    has_equipped_kitchen = st.selectbox("Equipped Kitchen?", [0, 1])

st.markdown("---")


# ---------------------------------------------------------
# METADATA (User Must Provide From Your Tables)
# ---------------------------------------------------------
st.subheader("📊 Location-Based Metadata (from your tables)")

st.caption("Provide TE/benchmark values as computed by your preprocessing tables.")

colA, colB = st.columns(2)

with colA:
    postal_code_te_price = st.number_input("postal_code_te_price", min_value=0, value=400000)
    locality_te_price = st.number_input("locality_te_price", min_value=0, value=390000)
    property_subtype_te_price = st.number_input("property_subtype_te_price", min_value=0, value=380000)
    property_type_te_price = st.number_input("property_type_te_price", min_value=0, value=370000)

with colB:
    province_benchmark_m2 = st.number_input("province_benchmark_m2", min_value=0, value=3000)
    region_benchmark_m2 = st.number_input("region_benchmark_m2", min_value=0, value=2800)
    apt_avg_m2_region = st.number_input("apt_avg_m2_region", min_value=0, value=3100)
    median_income = st.number_input("median_income", min_value=0, value=35000)

colC, colD = st.columns(2)

with colC:
    house_avg_m2_province = st.number_input("house_avg_m2_province", min_value=0, value=3200)
    apt_avg_m2_province = st.number_input("apt_avg_m2_province", min_value=0, value=3000)

with colD:
    house_avg_m2_region = st.number_input("house_avg_m2_region", min_value=0, value=2800)


st.markdown("---")


# ---------------------------------------------------------
# BUILD FINAL DICTIONARY FOR PREDICTION
# ---------------------------------------------------------
input_dict = {
    "property_type": property_type,
    "area": area,
    "bathrooms": bathrooms,
    "rooms": rooms,
    "primary_energy_consumption": primary_energy_consumption,
    "state": state,
    "postal_code": postal_code,
    "toilets": toilets,
    "build_year": build_year,
    "has_garage": has_garage,
    "has_garden": has_garden,
    "has_terrace": has_terrace,
    "facades_number": facades_number,
    "has_swimming_pool": has_swimming_pool,
    "has_equipped_kitchen": has_equipped_kitchen,

    # Metadata
    "postal_code_te_price": postal_code_te_price,
    "locality_te_price": locality_te_price,
    "province_benchmark_m2": province_benchmark_m2,
    "region_benchmark_m2": region_benchmark_m2,
    "property_subtype_te_price": property_subtype_te_price,
    "apt_avg_m2_region": apt_avg_m2_region,
    "property_type_te_price": property_type_te_price,
    "median_income": median_income,
    "house_avg_m2_province": house_avg_m2_province,
    "apt_avg_m2_province": apt_avg_m2_province,
    "house_avg_m2_region": house_avg_m2_region,
}


# ---------------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------------
if st.button("🔮 Predict Price"):
    with st.spinner("Computing price using ML model..."):
        predicted_price = predict_price(input_dict)

    st.success(f"🏷️ Estimated Market Price: **€{predicted_price:,.0f}**")


st.markdown("---")
st.caption("Immo-Eliza ML Price Predictor — Powered by XGBoost + Stage3 Preprocessing")

import streamlit as st
import numpy as np
import pandas as pd
import joblib

lr_model = joblib.load("linear_model(4).pkl")
svr_model = joblib.load("svr_model(3).pkl")
rf_model = joblib.load("rf_model(3).pkl")

st.title("Concrete Strength Predictor")

cement = st.number_input("Cement KG")
flyash = st.number_input("Fly Ash KG")
ggbs = st.number_input("GGBS KG")
silica = st.number_input("Silica Fume KG")
water = st.number_input("Water KG")
if cement > 0:
    wc_ratio = water / cement
    st.write("Water-Cement Ratio:", round(wc_ratio, 3))
else:
    wc_ratio = 0
    st.write("Water-Cement Ratio: Waiting for cement value")
fa = st.number_input("Fine Aggregate KG")
ca = st.number_input("Coarse Aggregate KG")
sp = st.number_input("Superplasticizer")
# Material Rates (₹ per kg)
cement_rate = 7.5
flyash_rate = 1.5      # you can assume ~₹1–2/kg
ggbs_rate = 3
silica_rate = 20
fine_agg_rate = 1.2
coarse_agg_rate = 1.1
sp_rate = 50           # ₹ per litre (approx)
if st.button("Predict Strength"):

    sample = pd.DataFrame([[cement, flyash, ggbs, silica, water, wc_ratio, fa, ca, sp]],
    columns=[
        "Cement_kg",
        "Fly_Ash_kg",
        "GGBS_kg",
        "Silica_Fume_kg",
        "Water_kg",
        "W_C_ratio",
        "Fine_Aggregate_kg",
        "Coarse_Aggregate_kg",
        "Superplasticizer_L"
    ])

    lr = lr_model.predict(sample)[0]
    svr = svr_model.predict(sample)[0]
    rf = rf_model.predict(sample)[0]

    st.write("Linear Regression:", round(lr,2), "MPa")
    st.write("SVR:", round(svr,2), "MPa")
    st.write("Random Forest:", round(rf,2), "MPa")

    # ✅ COST CALCULATION (fixed)
    cost = (
        cement * cement_rate +
        flyash * flyash_rate +
        ggbs * ggbs_rate +
        silica * silica_rate +
        fa * fine_agg_rate +
        ca * coarse_agg_rate +
        sp * sp_rate
    )

    st.subheader("💰 Cost Analysis")
    st.write(f"Total Cost per m³: ₹ {round(cost,2)}")
git add .
git commit -m "added cost analysis"
git push

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

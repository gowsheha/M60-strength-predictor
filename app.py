import streamlit as st
import numpy as np
import joblib

lr_model = joblib.load("linear_model.pkl")
svr_model = joblib.load("svr_model.pkl")
rf_model = joblib.load("rf_model.pkl")

st.title("Concrete Strength Predictor")

cement = st.number_input("Cement")
ggbs = st.number_input("GGBS")
silica = st.number_input("Silica Fume")
flyash = st.number_input("Fly Ash")
water = st.number_input("Water")
sp = st.number_input("Superplasticizer")
fa = st.number_input("Fine Aggregate")
ca = st.number_input("Coarse Aggregate")

if st.button("Predict Strength"):
    
    sample = np.array([[cement, ggbs, silica, flyash, water, sp, fa, ca]])
    
    lr = lr_model.predict(sample)[0]
    svr = svr_model.predict(sample)[0]
    rf = rf_model.predict(sample)[0]

    st.write("Linear Regression:", lr)
    st.write("SVR:", svr)
    st.write("Random Forest:", rf)

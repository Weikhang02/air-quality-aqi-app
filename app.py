import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# LOAD MODEL
# -----------------------------
artifact = joblib.load("rf_aqi_model.pkl")
model = artifact["model"]
features = artifact["features"]  # ['PM10','CO',"PM2.5",...]

st.title("AQI Prediction App (Random Forest)")
st.write("Predict next-day AQI based on pollutant levels.")

# -----------------------------
# USER INPUTS
# -----------------------------
st.sidebar.header("Input pollutant values")

user_input = {}
for feat in features:
    user_input[feat] = st.sidebar.number_input(
        feat,
        value=0.0,
        format="%.3f"
    )

if st.button("Predict AQI for 1 day ahead"):
    # Turn inputs into a DataFrame with correct column order
    X_new = pd.DataFrame([user_input])[features]
    pred = model.predict(X_new)[0]
    st.success(f"Predicted AQI (next day): {pred:.2f}")

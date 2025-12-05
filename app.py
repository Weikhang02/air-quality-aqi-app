import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("rf_aqi_model.pkl")   # this is now the real RF model
features = ['PM10','CO',"PM2.5",'NO2','SO2','NOx','NO','Toluene','NH3','O3']

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
    X_new = pd.DataFrame([user_input])[features]
    pred = model.predict(X_new)[0]
    st.success(f"Predicted AQI (next day): {pred:.2f}")

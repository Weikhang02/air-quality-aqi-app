import numpy as np
import pickle
import streamlit as st

@st.cache_resource
def load_model():
    with open("rf_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_feature_names():
    try:
        with open("rf_model_features.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return ['PM10','CO',"PM2.5",'NO2','SO2','NOx','NO','Toluene','NH3','O3']

model = load_model()
FEATURES = load_feature_names()

st.title("Next-Day AQI Prediction (Random Forest)")
st.write("Enter pollutant levels to predict tomorrow's AQI (AQI_next).")

st.sidebar.header("Input pollutant values")
inputs = {}
for feat in FEATURES:
    inputs[feat] = st.sidebar.number_input(feat, value=50.0, step=1.0)

if st.button("Predict next-day AQI"):
    X_new = np.array([[inputs[f] for f in FEATURES]], dtype=float)
    pred = model.predict(X_new)[0]

    st.subheader("Prediction")
    st.metric("Predicted next-day AQI", f"{pred:.2f}")

    if pred <= 50:
        category = "Good"
    elif pred <= 100:
        category = "Satisfactory"
    elif pred <= 200:
        category = "Moderate"
    elif pred <= 300:
        category = "Poor"
    elif pred <= 400:
        category = "Very Poor"
    else:
        category = "Severe"

    st.write(f"**AQI Category:** {category}")

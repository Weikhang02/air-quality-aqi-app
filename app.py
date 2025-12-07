import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# LOAD MODEL + FEATURES + TARGETS
# -----------------------------
model = joblib.load("rf_aqi_8output_model.pkl")
feature_cols = joblib.load("aqi_feature_cols.pkl")
target_cols = joblib.load("aqi_target_cols.pkl")

st.title("AQI Forecasting App (Today + 7 Days)")
st.write("Enter today's pollutant values to predict **current AQI** and **AQI for the next 7 days**.")

# -----------------------------
# SIDEBAR USER INPUT FOR POLLUTANTS
# -----------------------------
st.sidebar.header("Enter pollutant values")

user_input = {}
for feat in feature_cols:
    user_input[feat] = st.sidebar.number_input(
        feat,
        value=0.0,
        format="%.3f"
    )

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("Predict AQI (Today + 7 Days)"):
    
    # Convert → DataFrame (1 row)
    X_new = pd.DataFrame([user_input], columns=feature_cols)

    # Predict → shape (1, 8)
    preds = model.predict(X_new)[0]

    # Prepare output table
    result_df = pd.DataFrame({
        "Day": [
            "Today (AQI)",
            "Day +1",
            "Day +2",
            "Day +3",
            "Day +4",
            "Day +5",
            "Day +6",
            "Day +7"
        ],
        "Predicted_AQI": preds
    })

    st.subheader("AQI Prediction Results")
    st.table(result_df)

import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("model.pkl")

st.title("🍽️ Hotel F&B Demand Predictor")
st.write("Predict food demand based on inputs like season, events, etc.")

# Input fields
guest_count = st.number_input("Guest Count", min_value=0)
temperature = st.number_input("Temperature (°C)")
event = st.selectbox("Is there a major event?", ["No", "Yes"])

# Convert inputs to a DataFrame
event_encoded = 1 if event == "Yes" else 0
input_df = pd.DataFrame([[guest_count, temperature, event_encoded]],
                        columns=["guest_count", "temperature", "event_flag"])

# Predict and display
if st.button("Predict Demand"):
    prediction = model.predict(input_df)
    st.success(f"Predicted demand: {prediction[0]:.2f} units")

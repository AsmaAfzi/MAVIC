import streamlit as st
import pandas as pd
import joblib
model = joblib.load("model.pkl")


st.title("📄 Upload CSV for Analysis")

# Upload CSV
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read and display CSV
    df = pd.read_csv(uploaded_file)
    st.success("CSV uploaded successfully!")
    
    st.subheader("📊 Preview of Uploaded Data")
    st.dataframe(df)

    # (Optional) Use the CSV for predictions or analysis
    # Example: st.write(model.predict(df[model_features]))

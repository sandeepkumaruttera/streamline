import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

st.set_page_config(layout="wide")
st.title("🧠 ML Dashboard with Visualizations")

# Try to load the model (if present)
@st.cache_resource
def load_model():
    model_path = "model.pkl"  # You can rename your file accordingly
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

def get_predictions(data):
    if model is not None:
        try:
            return model.predict(data)
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            return None
    else:
        st.warning("⚠️ No model file found. Upload a model.pkl to enable predictions.")
        return None

# File uploader
uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("✅ File loaded successfully!")
        st.write("### 🔍 Preview of Data", data.head())

        # Handle numeric data
        numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()

        if numeric_cols:
            st.subheader("📈 Line Chart (first 100 rows)")
            st.line_chart(data[numeric_cols].head(100))

            st.subheader("📊 Correlation Heatmap")
            corr = data[numeric_cols].corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            st.subheader("📌 Histograms")
            for col in numeric_cols:
                fig, ax = plt.subplots()
                sns.histplot(data[col], kde=True, ax=ax)
                ax.set_title(f"Histogram of {col}")
                st.pyplot(fig)
        else:
            st.info("No numeric columns found for visualization.")

        # Make predictions
        st.subheader("🤖 Predictions")
        predictions = get_predictions(data)
        if predictions is not None:
            st.write("### 📋 Model Predictions")
            st.write(predictions)

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("📎 Please upload a CSV file to begin.")

# app.py - Streamlit app for Breast Cancer Prediction (manual + CSV)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")

st.title("🩺 Breast Cancer Prediction")
st.markdown(
    "This app predicts **Benign (0)** or **Malignant (1)** from tumor features. "
    "You can either enter values manually (Single) or upload a CSV (Batch)."
)

# ---------------------------
# Load model & scaler
# ---------------------------
@st.cache_resource
def load_models():
    model = joblib.load("breast_cancer_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_models()
except Exception as e:
    st.error(f"Failed to load model/scaler. Make sure `breast_cancer_model.pkl` and `scaler.pkl` are in this folder. Error: {e}")
    st.stop()

# ---------------------------
# Expected feature names (Kaggle Breast Cancer dataset - 30 features)
# IMPORTANT: Order must match the order used during training
# ---------------------------
FEATURE_NAMES = [
 "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
 "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean",
 "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
 "compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se",
 "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
 "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"
]

# ---------------------------
# Helper functions
# ---------------------------
def predict_single(input_list):
    arr = np.array(input_list).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    pred = model.predict(arr_scaled)[0]
    pred_proba = None
    if hasattr(model, "predict_proba"):
        pred_proba = model.predict_proba(arr_scaled)[0].max()
    return int(pred), float(pred_proba) if pred_proba is not None else None

def predict_batch(df):
    # ensure columns ordering
    X = df[FEATURE_NAMES].values
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled).max(axis=1)
    return preds, proba

# ---------------------------
# Sidebar: mode selection
# ---------------------------
st.sidebar.header("Mode")
mode = st.sidebar.radio("Choose input mode", ["Single Input", "Batch CSV"])

# Use scaler mean as defaults if possible
defaults = None
if hasattr(scaler, "mean_"):
    defaults = list(scaler.mean_)
else:
    defaults = [0.0]*len(FEATURE_NAMES)

# ---------------------------
# SINGLE INPUT
# ---------------------------
if mode == "Single Input":
    st.header("Single patient input")
    st.markdown("Enter values for each feature (you can paste numbers). Defaults come from training data mean.")

    # Use a form to avoid auto reruns
    with st.form("single_form"):
        inputs = []
        cols = st.columns(2)
        for i, fname in enumerate(FEATURE_NAMES):
            col = cols[i % 2]
            default_val = float(defaults[i]) if defaults is not None else 0.0
            # small width number input
            val = col.number_input(label=fname, value=round(default_val, 4), format="%.6f", key=f"f{i}")
            inputs.append(val)

        submitted = st.form_submit_button("Predict")
        if submitted:
            try:
                pred, proba = predict_single(inputs)
                if pred == 1:
                    st.error("⚠️ Prediction: Malignant (1)")
                else:
                    st.success("✅ Prediction: Benign (0)")
                if proba is not None:
                    st.write(f"Model confidence (max prob): {proba:.3f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ---------------------------
# BATCH CSV Upload
# ---------------------------
else:
    st.header("Batch predictions via CSV upload")
    st.markdown("Upload a CSV file containing the 30 features as columns. Column names must match the expected feature names exactly.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            missing = [c for c in FEATURE_NAMES if c not in df.columns]
            if missing:
                st.error(f"CSV is missing required columns. Missing: {missing}")
            else:
                st.success("CSV looks good — running predictions...")
                preds, proba = predict_batch(df)
                df_out = df.copy()
                df_out["prediction"] = preds
                if proba is not None:
                    df_out["confidence"] = proba
                st.dataframe(df_out.head(200))

                # download button
                to_download = df_out.copy()
                csv_bytes = to_download.to_csv(index=False).encode('utf-8')
                st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Failed to read/process CSV: {e}")

# ---------------------------
# Footer / Notes
# ---------------------------
st.markdown("---")
st.write("Notes:")
st.write("- The model was trained on the Kaggle Breast Cancer dataset. Make sure CSV column names and ordering match.")
st.write("- This app predicts based on numeric biopsy features, NOT medical images.")
st.write("- For image-based prediction (scans), we need a CNN model and image dataset — we can add that next.")

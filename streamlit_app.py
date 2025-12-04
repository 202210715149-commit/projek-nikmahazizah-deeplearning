import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ======================================================
# LOAD MODEL & COLUMN INFO
# ======================================================

st.set_page_config(page_title="Prediksi Performa Akademik Mahasiswa",
                   page_icon="🎓")

st.title("🎓 Prediksi Performa Akademik Mahasiswa")
st.write("Menggunakan Random Forest untuk Prediksi CGPA dan K-Means untuk Segmentasi Pola Belajar.")

# Load model dan metadata
rf_model = joblib.load("rf_pipeline.pkl")
kmeans_model = joblib.load("kmeans_pipeline.pkl")

with open("column_info.json", "r", encoding="utf-8") as f:
    colinfo = json.load(f)

feature_cols = colinfo["feature_columns"]
numeric_features = colinfo["numeric_features"]
categorical_features = colinfo["categorical_features"]
cluster_features = colinfo["cluster_features"]

# ======================================================
# FORM INPUT STREAMLIT
# ======================================================

st.header("📝 Input Data Mahasiswa")

user_input = {}

for col in feature_cols:
    if col in numeric_features:
        user_input[col] = st.number_input(col, value=0.0)
    else:
        user_input[col] = st.text_input(col, "")

# Convert menjadi DataFrame
input_df = pd.DataFrame([user_input])

# ======================================================
# PREDIKSI
# ======================================================

if st.button("🔮 Prediksi Sekarang"):
    # Prediksi CGPA
    pred_cgpa = rf_model.predict(input_df)[0]

    st.subheader("📘 Hasil Prediksi CGPA")
    st.success(f"Prediksi CGPA: **{pred_cgpa:.2f}**")

    # Prediksi Lulus / Tidak
    threshold = 2.75  # nilai dapat diganti sesuai aturan kampus
    status = "LULUS" if pred_cgpa >= threshold else "TIDAK LULUS"

    st.subheader("🏁 Status Kelulusan")
    if status == "LULUS":
        st.success("✔ Mahasiswa diprediksi **LULUS**")
    else:
        st.error("✘ Mahasiswa diprediksi **TIDAK LULUS**")

    st.caption(f"Ambang batas kelulusan: CGPA ≥ {threshold}")

    # Prediksi Cluster
    cluster_input = input_df[cluster_features]
    cluster_label = kmeans_model.predict(cluster_input)[0]

    st.subheader("📊 Segmentasi Pola Belajar (Cluster)")
    st.info(f"Mahasiswa masuk **Cluster {cluster_label}**")

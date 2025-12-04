streamlit_app_code = r'''import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(page_title="Student Performance App", layout="wide")

st.title("📊 Student Performance Prediction & Clustering")

# Load model & metadata
@st.cache_resource
def load_models():
    rf = joblib.load("rf_pipeline.pkl")
    kmeans = joblib.load("kmeans_pipeline.pkl")
    with open("column_info.json", "r") as f:
        col_info = json.load(f)
    return rf, kmeans, col_info

rf_pipeline, kmeans_pipeline, col_info = load_models()

FEATURE_COLUMNS = col_info["feature_columns"]
CLUSTER_FEATURES = col_info["cluster_features"]
TARGET_COL = col_info["target"]

st.sidebar.header("Mode Aplikasi")
mode = st.sidebar.radio(
    "Pilih mode:",
    ["Info", "Upload CSV & Prediksi", "Contoh Format Input"]
)

if mode == "Info":
    st.write("""
    Aplikasi ini menggunakan:
    - **Random Forest Regressor** untuk memprediksi *Current CGPA* mahasiswa
    - **K-Means Clustering** untuk mengelompokkan mahasiswa berdasarkan:
      - Jam belajar harian
      - Frekuensi duduk belajar per hari
      - SKS yang sudah ditempuh
      - Umur
      - Pendapatan keluarga per bulan

    ### Cara pakai
    1. Siapkan file CSV dengan struktur kolom yang sama seperti dataset training.
    2. Buka tab **Upload CSV & Prediksi**.
    3. Upload file, lalu klik tombol prediksi.
    """)
elif mode == "Contoh Format Input":
    st.write("Berikut contoh nama-nama kolom yang dibutuhkan model:")
    st.write("**Target (diprediksi):**")
    st.code(TARGET_COL)
    st.write("**Fitur (X):**")
    st.write(FEATURE_COLUMNS)
    st.write("**Fitur yang digunakan untuk Clustering (K-Means):**")
    st.write(CLUSTER_FEATURES)
else:
    st.header("📁 Upload CSV Mahasiswa")
    uploaded_file = st.file_uploader("Upload file CSV dengan kolom yang sama seperti dataset training", type=["csv"])

    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.subheader("Preview Data")
        st.dataframe(df_input.head())

        # Cek apakah semua kolom fitur ada
        missing_cols = [c for c in FEATURE_COLUMNS if c not in df_input.columns]
        if missing_cols:
            st.error(f"Kolom berikut belum ada di file yang diupload: {missing_cols}")
        else:
            if st.button("🚀 Jalankan Prediksi & Clustering"):
                # Pastikan urutan kolom sesuai
                X_input = df_input[FEATURE_COLUMNS].copy()

                # Prediksi CGPA
                pred_cgpa = rf_pipeline.predict(X_input)

                # Prediksi cluster (gunakan CLUSTER_FEATURES)
                X_cluster = df_input[CLUSTER_FEATURES].copy()
                cluster_labels = kmeans_pipeline.predict(X_cluster)

                # Tambahkan ke dataframe
                result_df = df_input.copy()
                result_df["Predicted_CGPA"] = pred_cgpa
                result_df["Cluster"] = cluster_labels

                st.subheader("Hasil Prediksi")
                st.dataframe(result_df.head())

                # Unduh hasil
                csv_out = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download Hasil dalam CSV",
                    data=csv_out,
                    file_name="prediction_result.csv",
                    mime="text/csv"
                )

                # Ringkasan cluster
                st.subheader("Ringkasan Rata-rata per Cluster")
                st.write(result_df.groupby("Cluster")[CLUSTER_FEATURES + ["Predicted_CGPA"]].mean())
'''

with open("streamlit_app.py", "w") as f:
    f.write(streamlit_app_code)
print("Generated: streamlit_app.py")
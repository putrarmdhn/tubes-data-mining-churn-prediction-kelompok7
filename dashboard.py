import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings('ignore')

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Bank Churn Dashboard", page_icon="🏦", layout="wide")

# Fungsi untuk memuat model dan preprocessor
@st.cache_resource
def load_resources():
    try:
        scaler_class = joblib.load('scaler_class.pkl')
        scaler_cluster = joblib.load('scaler_cluster.pkl')
        le_geo = joblib.load('le_geo.pkl')
        le_gender = joblib.load('le_gender.pkl')
        kmeans = joblib.load('kmeans_model.pkl')
        logreg = joblib.load('logreg_model.pkl')
        return scaler_class, scaler_cluster, le_geo, le_gender, kmeans, logreg
    except Exception as e:
        return None, None, None, None, None, None

scaler_class, scaler_cluster, le_geo, le_gender, kmeans, logreg = load_resources()

# Load Data untuk eksplorasi
@st.cache_data
def load_data():
    df = pd.read_csv('Churn_Modelling.csv')
    return df

df = load_data()

# Sidebar Navigasi
st.sidebar.title("Menu Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", ["Data Exploration", "Churn Prediction"])

st.sidebar.markdown("---")
st.sidebar.info("Tugas Besar Data Mining - Kelompok 7")

if menu == "Data Exploration":
    st.title(" Data Exploration & Business Insights")
    st.markdown("Dashboard interaktif untuk eksplorasi dataset **Customer Churn**. Tujuan utama dari halaman ini adalah untuk memahami karakteristik nasabah dan menemukan faktor apa saja yang mempengaruhi keputusan mereka untuk meninggalkan bank (Churn).")
    
    # KPI Section
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Nasabah", f"{len(df):,}")
    col2.metric("Total Churn (Keluar)", f"{df['Exited'].sum():,}")
    churn_rate = (df['Exited'].sum() / len(df)) * 100
    col3.metric("Churn Rate (%)", f"{churn_rate:.2f}%")
    
    st.markdown("---")
    
    # Visualisasi 1: Target Variable
    st.subheader("1. Distribusi Status Nasabah (Target Variable)")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x='Exited', data=df, palette='Set2', ax=ax)
    ax.set_xticklabels(['Stay (0)', 'Churn (1)'])
    ax.set_xlabel("Status Nasabah")
    ax.set_ylabel("Jumlah")
    st.pyplot(fig)
    st.markdown("> **Insight:** Dataset mengalami ketidakseimbangan kelas (*imbalanced data*), di mana jumlah nasabah yang bertahan jauh lebih banyak dibandingkan yang keluar. Hal ini perlu diperhatikan saat mengevaluasi model klasifikasi.")
    
    st.markdown("---")
    
    # Visualisasi 2: Distribusi Fitur Numerik
    st.subheader("2. Analisis Distribusi Fitur Numerik")
    feature = st.selectbox("Pilih Variabel Numerik untuk Dilihat Distribusinya:", 
                           ['Age', 'Balance', 'CreditScore', 'EstimatedSalary'])
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.histplot(df, x=feature, hue="Exited", kde=True, palette='magma', ax=ax2, bins=30)
    ax2.set_title(f"Distribusi {feature} Berdasarkan Status Churn")
    st.pyplot(fig2)
    st.markdown(f"> **Insight Eksplorasi {feature}:** Grafik di atas menunjukkan bagaimana kelompok nasabah yang churn vs stay terdistribusi pada metrik **{feature}**.")
    
    st.markdown("---")
    
    # Visualisasi 3: Correlation Heatmap
    st.subheader("3. Heatmap Korelasi Antar Fitur")
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    numeric_df = df.select_dtypes(include=['int64', 'float64']).drop(columns=['RowNumber', 'CustomerId'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax3)
    ax3.set_title("Korelasi Variabel Numerik")
    st.pyplot(fig3)
    st.markdown("> **Insight Korelasi:** Variabel `Age` memiliki korelasi positif yang paling tinggi terhadap target `Exited` dibandingkan variabel lainnya. Ini mengindikasikan bahwa nasabah yang lebih tua cenderung memiliki risiko churn yang lebih tinggi.")

elif menu == "Churn Prediction":
    st.title("Customer Churn Prediction")
    st.markdown("Halaman ini merupakan implementasi tahapan **Deployment**. Anda dapat memasukkan profil data nasabah baru untuk memprediksi apakah nasabah tersebut memiliki kecenderungan untuk Churn atau tidak.")
    
    if logreg is None:
        st.error("⚠️ Model belum dimuat atau tidak ditemukan! Pastikan Anda telah menjalankan notebook/script training untuk menghasilkan file model (contoh: logreg_model.pkl).")
    else:
        st.markdown("### Masukkan Data Nasabah Baru")
        with st.form("predict_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                credit_score = st.number_input("Credit Score (Skor Kredit)", min_value=300, max_value=900, value=600)
                geography = st.selectbox("Geography (Negara)", options=["France", "Germany", "Spain"])
                gender = st.selectbox("Gender (Jenis Kelamin)", options=["Female", "Male"])
                age = st.number_input("Age (Umur)", min_value=18, max_value=100, value=40)
                tenure = st.number_input("Tenure (Lama Menjadi Nasabah dalam Tahun)", min_value=0, max_value=10, value=3)
                
            with col2:
                balance = st.number_input("Balance (Saldo di Akun)", min_value=0.0, value=60000.0, step=1000.0)
                num_products = st.number_input("Number of Products (Jumlah Produk Bank yang Dimiliki)", min_value=1, max_value=4, value=2)
                has_cr_card = st.selectbox("Has Credit Card? (Memiliki Kartu Kredit?)", options=[1, 0], format_func=lambda x: "Yes" if x==1 else "No")
                is_active = st.selectbox("Is Active Member? (Aktif Bertransaksi?)", options=[1, 0], format_func=lambda x: "Yes" if x==1 else "No")
                est_salary = st.number_input("Estimated Salary (Estimasi Gaji)", min_value=0.0, value=50000.0, step=1000.0)
                
            submit_btn = st.form_submit_button("Lakukan Prediksi")
            
        if submit_btn:
            # 1. Siapkan data sesuai inputan form
            new_data = {
                'CreditScore': credit_score,
                'Geography': geography,
                'Gender': gender,
                'Age': age,
                'Tenure': tenure,
                'Balance': balance,
                'NumOfProducts': num_products,
                'HasCrCard': has_cr_card,
                'IsActiveMember': is_active,
                'EstimatedSalary': est_salary
            }
            
            df_new = pd.DataFrame([new_data])
            
            # 2. Transformasi Kategorikal (menggunakan encoder dari training)
            df_new['Geography'] = le_geo.transform(df_new['Geography'])
            df_new['Gender'] = le_gender.transform(df_new['Gender'])
            
            # 3. Clustering (Menentukan segmen nasabah menggunakan K-Means model)
            cluster_features = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
            X_cluster = df_new[cluster_features]
            X_cluster_scaled = scaler_cluster.transform(X_cluster)
            df_new['Cluster'] = kmeans.predict(X_cluster_scaled)
            
            # 4. Susun urutan kolom (fitur) agar persis seperti yang digunakan classifier
            feature_order = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 
                             'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Cluster']
            df_new = df_new[feature_order]
            
            # 5. Scaling untuk Model Supervised
            df_new_scaled = scaler_class.transform(df_new)
            
            # 6. Lakukan Prediksi (Logistic Regression)
            pred = logreg.predict(df_new_scaled)
            prob = logreg.predict_proba(df_new_scaled)[0][1]
            
            # 7. Tampilkan Hasil
            st.markdown("---")
            st.markdown("### Hasil Prediksi")
            
            if pred[0] == 1:
                st.error(f"🚨 **Peringatan!** Nasabah ini diprediksi **CHURN (Meninggalkan Bank)** dengan probabilitas sebesar **{prob*100:.2f}%**.")
            else:
                st.success(f"✅ **Aman!** Nasabah ini diprediksi **STAY (Tetap di Bank)** dengan probabilitas churn hanya sebesar **{prob*100:.2f}%**.")
                
            st.info("Catatan: Prediksi ini menggunakan model Logistic Regression yang telah dilatih pada tahap pembuatan model, dan fitur tambahan 'Cluster' dari hasil segmentasi Unsupervised K-Means turut disertakan dalam proses prediksi.")
import json

notebook = {
    "cells": [],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_markdown(source):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split('\n')]
    })

def add_code(source):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split('\n')]
    })

add_markdown("# TUGAS BESAR PENAMBANGAN DATA\n\n**Analisis Segmentasi Karakteristik Nasabah dan Prediksi Customer Churn**\n\nBerdasarkan Dataset Churn_Modelling")

add_markdown("## 1. Pendahuluan & Import Library\n\nMempersiapkan environment dengan mengimpor library-library yang penting untuk Data Wrangling, Visualisasi Data, dan Machine Learning.")
add_code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # Untuk menyimpan model deployment

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

import warnings
warnings.filterwarnings('ignore')

# Set visual style
sns.set_theme(style="whitegrid")""")

add_markdown("## 2. Data Understanding & Exploration\n\nMembaca dataset, mengecek tipe data, mendeteksi nilai yang hilang, dan melakukan **Eksplorasi Data (EDA)** secara menyeluruh untuk mendapatkan Business Insight.")
add_code("""# Load dataset
df = pd.read_csv('Churn_Modelling.csv')
display(df.head())""")

add_code("""# Info dataset dan tipe data
df.info()""")

add_code("""# Pengecekan missing values
print("Missing values di setiap kolom:")
display(df.isnull().sum())

# Cek Duplikasi
print(f"Jumlah duplikasi data: {df.duplicated().sum()}")""")

add_code("""# Statistik deskriptif
display(df.describe())""")

add_markdown("### 2.1 Analisis Distribusi Target (Churn)")
add_code("""plt.figure(figsize=(6, 4))
sns.countplot(x='Exited', data=df, palette='viridis')
plt.title('Distribusi Customer Churn (0: Stay, 1: Exited)')
plt.show()

churn_rate = df['Exited'].value_counts(normalize=True) * 100
print(f"Persentase Churn: {churn_rate[1]:.2f}%")
print(f"Persentase Stay: {churn_rate[0]:.2f}%")
# Insight: Dataset tidak seimbang (imbalanced), nasabah yang stay lebih banyak dibandingkan yang churn.""")

add_markdown("### 2.2 Analisis Karakteristik Numerik (Age, Balance, Credit Score)")
add_code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(df['Age'], bins=30, kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Distribusi Umur Nasabah')

sns.histplot(df['Balance'], bins=30, kde=True, ax=axes[1], color='salmon')
axes[1].set_title('Distribusi Saldo (Balance)')

sns.histplot(df['CreditScore'], bins=30, kde=True, ax=axes[2], color='lightgreen')
axes[2].set_title('Distribusi Credit Score')

plt.tight_layout()
plt.show()""")

add_markdown("### 2.3 Analisis Bivariat (Korelasi Fitur dengan Churn)")
add_code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(x='Exited', y='Age', data=df, ax=axes[0], palette='Set2')
axes[0].set_title('Umur vs Churn')

sns.boxplot(x='Exited', y='Balance', data=df, ax=axes[1], palette='Set2')
axes[1].set_title('Saldo vs Churn')

plt.show()
# Insight: Nasabah yang keluar (churn) cenderung memiliki umur yang lebih tua (median lebih tinggi).""")

add_markdown("### 2.4 Korelasi Heatmap")
add_code("""plt.figure(figsize=(10, 8))
# Ambil hanya kolom numerik untuk korelasi
numeric_cols = df.select_dtypes(include=['int64', 'float64']).drop(columns=['RowNumber', 'CustomerId'])
corr = numeric_cols.corr()

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Heatmap Korelasi Antar Variabel Numerik')
plt.show()
# Insight: Age memiliki korelasi positif paling tinggi terhadap Exited dibandingkan fitur numerik lainnya.""")

add_markdown("## 3. Data Preparation\n\nMelakukan *Data Cleansing* dan Transformasi Data agar algoritma Machine Learning dapat bekerja dengan optimal.")
add_code("""# Drop kolom yang tidak relevan (Identitas unik yang tidak memiliki pola)
df_clean = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# Encoding categorical variables (Geography, Gender) menggunakan LabelEncoder
le_geo = LabelEncoder()
df_clean['Geography'] = le_geo.fit_transform(df_clean['Geography'])
# Simpan mapping: France=0, Germany=1, Spain=2

le_gender = LabelEncoder()
df_clean['Gender'] = le_gender.fit_transform(df_clean['Gender'])
# Simpan mapping: Female=0, Male=1

display(df_clean.head())""")

add_markdown("## 4. Unsupervised Learning - K-Means Clustering\n\nMelakukan segmentasi nasabah secara *unsupervised* untuk melihat pola/grup nasabah yang ada di bank.")
add_code("""# Memilih fitur untuk clustering (Segmentasi Finansial & Demografis)
cluster_features = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
X_cluster = df_clean[cluster_features]

# Scaling data khusus untuk K-Means
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

# K-Means Clustering dengan k=3 (Misal kita asumsikan 3 segmen utama)
kmeans = KMeans(n_clusters=3, random_state=42)
df_clean['Cluster'] = kmeans.fit_predict(X_cluster_scaled)

print("Jumlah nasabah per cluster:")
print(df_clean['Cluster'].value_counts())

# Menampilkan karakteristik per cluster
display(df_clean.groupby('Cluster')[cluster_features].mean())
# Insight: Profiling nasabah bisa dilakukan berdasar nilai rata-rata tiap fitur dalam cluster.""")

add_markdown("## 5. Supervised Learning - Classification\n\nMemprediksi probabilitas *Customer Churn* menggunakan model *Logistic Regression* dan *Naïve Bayes*.")
add_code("""# Menyiapkan variabel dependen (target) dan independen (fitur)
# Kita drop target 'Exited'
X = df_clean.drop('Exited', axis=1)
y = df_clean['Exited']

# Split data (80% Train, 20% Test) menggunakan stratify agar proporsi churn di train dan test seimbang
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling seluruh fitur untuk klasifikasi
scaler_class = StandardScaler()
X_train_scaled = scaler_class.fit_transform(X_train)
X_test_scaled = scaler_class.transform(X_test)""")

add_markdown("### 5.1 Logistic Regression")
add_code("""logreg = LogisticRegression(random_state=42, class_weight='balanced')
logreg.fit(X_train_scaled, y_train)

y_pred_lr = logreg.predict(X_test_scaled)
print("--- LOGISTIC REGRESSION ---")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\\nClassification Report:\\n", classification_report(y_test, y_pred_lr))

# Visualisasi Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()""")

add_markdown("### 5.2 Naïve Bayes Classifier")
add_code("""nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

y_pred_nb = nb_model.predict(X_test_scaled)
print("--- NAÏVE BAYES ---")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\\nClassification Report:\\n", classification_report(y_test, y_pred_nb))

# Visualisasi Confusion Matrix
cm_nb = confusion_matrix(y_test, y_pred_nb)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix - Naive Bayes')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()""")

add_markdown("## 6. Export Model untuk Deployment\n\nMenyimpan (serialize) objek preprocessor dan model terbaik menggunakan `joblib` agar dapat digunakan dalam aplikasi Dashboard Streamlit.")
add_code("""# Menyimpan Scaler dan Encoder
joblib.dump(scaler_class, 'scaler_class.pkl')
joblib.dump(scaler_cluster, 'scaler_cluster.pkl')
joblib.dump(le_geo, 'le_geo.pkl')
joblib.dump(le_gender, 'le_gender.pkl')

# Menyimpan Model Machine Learning
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(logreg, 'logreg_model.pkl') # Kita gunakan Logistic Regression untuk prediksi utama
joblib.dump(nb_model, 'nb_model.pkl')

print("Semua objek preprocessing dan model telah berhasil disimpan ke format .pkl")""")

with open('Tubes_DataMining_Kelompok.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated.")

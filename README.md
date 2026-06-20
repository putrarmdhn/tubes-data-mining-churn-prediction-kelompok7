# Customer Churn Prediction Dashboard

## BBK2LAB3 - Penambangan Data

## Kelas SI4807

### Kelompok 7

- Putra Ramadhan - 102022580051
- Fitria Imanda Satriawan - 102022580049

## Deskripsi Proyek

Proyek ini bertujuan untuk menganalisis Customer Churn menggunakan teknik Data Mining.

### Metode yang Digunakan

#### Unsupervised Learning
- K-Means Clustering

#### Supervised Learning
- Logistic Regression
- Naive Bayes Classifier

## Dataset

Churn Modelling Dataset

## Tools

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Matplotlib
- Seaborn

## Fitur Dashboard

### Data Exploration
- Distribusi Customer Churn
- Analisis Variabel Numerik
- Heatmap Korelasi

### Customer Churn Prediction
- Prediksi Stay / Churn
- Probabilitas Churn
- Segmentasi Cluster Nasabah

## Dashboard

https://customer-churn-kelompok7.streamlit.app

### Features
- Data Exploration
- Customer Churn Prediction
- K-Means Clustering Segmentation
- Logistic Regression Classification
- Naive Bayes Comparison

## Repository

https://github.com/putrarmdhn/tubes-data-mining-churn-prediction-kelompok7

## Revision Note

Berdasarkan masukan dosen, dilakukan revisi pada bagian penanganan data imbalanced. Dataset Customer Churn memiliki distribusi kelas yang tidak seimbang, yaitu 79,63% nasabah tidak churn dan 20,37% nasabah churn.

Perbaikan yang dilakukan:

* Menambahkan eksperimen Logistic Regression sebelum penanganan imbalance.
* Menambahkan eksperimen Logistic Regression setelah penanganan imbalance menggunakan `class_weight='balanced'`.
* Membandingkan hasil sebelum dan sesudah penanganan imbalance berdasarkan Accuracy, Precision, Recall, dan F1-Score.
* Model yang digunakan pada dashboard adalah Logistic Regression setelah penanganan imbalance.

Hasil perbandingan:

| Model                                          | Accuracy | Precision Churn | Recall Churn | F1-Score Churn |
| ---------------------------------------------- | -------: | --------------: | -----------: | -------------: |
| Logistic Regression sebelum handling imbalance |    80.6% |             59% |          15% |            24% |
| Logistic Regression setelah handling imbalance |    70.3% |             38% |          71% |            49% |

Setelah penanganan imbalance menggunakan `class_weight='balanced'`, nilai recall churn meningkat dari 15% menjadi 71%. Hal ini menunjukkan bahwa model lebih mampu mendeteksi nasabah yang berpotensi churn, meskipun accuracy menurun.

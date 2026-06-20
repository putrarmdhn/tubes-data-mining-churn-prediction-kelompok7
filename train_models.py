import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading dataset...")
df = pd.read_csv("Churn_Modelling.csv")

print("Data Preparation...")
df_clean = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

le_geo = LabelEncoder()
df_clean["Geography"] = le_geo.fit_transform(df_clean["Geography"])

le_gender = LabelEncoder()
df_clean["Gender"] = le_gender.fit_transform(df_clean["Gender"])

print("Training K-Means...")
cluster_features = ["CreditScore", "Age", "Balance", "EstimatedSalary"]
X_cluster = df_clean[cluster_features]

scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42)
df_clean["Cluster"] = kmeans.fit_predict(X_cluster_scaled)

print("Preparing data for classification...")
X = df_clean.drop("Exited", axis=1)
y = df_clean["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler_class = StandardScaler()
X_train_scaled = scaler_class.fit_transform(X_train)
X_test_scaled = scaler_class.transform(X_test)

print("Training Classification Models...")

# ==================================================
# 1. Logistic Regression SEBELUM handling imbalance
# ==================================================
print("\n========================================")
print("LOGISTIC REGRESSION - SEBELUM HANDLING IMBALANCE")
print("========================================")

logreg_before = LogisticRegression(random_state=42)
logreg_before.fit(X_train_scaled, y_train)

y_pred_before = logreg_before.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred_before))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_before))
print("Classification Report:")
print(classification_report(y_test, y_pred_before))

# ==================================================
# 2. Logistic Regression SESUDAH handling imbalance
#    Teknik: class_weight='balanced'
# ==================================================
print("\n========================================")
print("LOGISTIC REGRESSION - SESUDAH HANDLING IMBALANCE")
print("========================================")

logreg_after = LogisticRegression(
    random_state=42,
    class_weight="balanced"
)

logreg_after.fit(X_train_scaled, y_train)

y_pred_after = logreg_after.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred_after))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_after))
print("Classification Report:")
print(classification_report(y_test, y_pred_after))

# ==================================================
# 3. Naive Bayes
# ==================================================
print("\n========================================")
print("NAIVE BAYES")
print("========================================")

nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

y_pred_nb = nb_model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))
print("Classification Report:")
print(classification_report(y_test, y_pred_nb))

print("Saving models to .pkl...")

joblib.dump(scaler_class, "scaler_class.pkl")
joblib.dump(scaler_cluster, "scaler_cluster.pkl")
joblib.dump(le_geo, "le_geo.pkl")
joblib.dump(le_gender, "le_gender.pkl")
joblib.dump(kmeans, "kmeans_model.pkl")

# Yang disimpan untuk dashboard adalah model SETELAH handling imbalance
joblib.dump(logreg_after, "logreg_model.pkl")
joblib.dump(nb_model, "nb_model.pkl")

print("All models successfully trained and saved!")

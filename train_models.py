# ============================================================
# ML Assignment 2 - Model Training
# Dataset: student_depression_dataset.csv
# Task: Student Depression Prediction
# ============================================================

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")


# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# Load Dataset
df = pd.read_csv("student_depression_dataset.csv")


# Target & Features
TARGET_COLUMN = "Depression"

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]


# Handle Missing Values
for col in X.columns:
    if X[col].dtype == "object":
        X[col].fillna(X[col].mode()[0], inplace=True)
    else:
        X[col].fillna(X[col].median(), inplace=True)


# Encode Categorical Variables
label_encoders = {}

for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le


# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


# Models Dictionary
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
}


# Evaluation Function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }


# Train, Evaluate & Save Models
results = []

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    metrics["Model"] = name
    results.append(metrics)

    # Save model
    file_name = name.replace(" ", "_") + ".pkl"
    joblib.dump(model, f"model/{file_name}")


# Save Preprocessing Objects
joblib.dump(
    {
        "scaler": scaler,
        "label_encoders": label_encoders
    },
    "model/preprocessor.pkl"
)


# Results Summary
results_df = pd.DataFrame(results)
results_df = results_df[
    ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
]

print("\n====================== Model Performance Comparison: ======================\n")
print(results_df.round(3))

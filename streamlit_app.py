import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)


# Page Setup
st.set_page_config(page_title="Student Depression Prediction", layout="centered")
st.title("🎓 Student Depression Prediction")
st.write("Upload test data, select a model, and view results.")


# Load Preprocessor and Models
preprocessor = joblib.load("model/preprocessor.pkl")
scaler = preprocessor["scaler"]
label_encoders = preprocessor["label_encoders"]

models = {
    "Logistic Regression": joblib.load("model/Logistic_Regression.pkl"),
    "Decision Tree": joblib.load("model/Decision_Tree.pkl"),
    "kNN": joblib.load("model/kNN.pkl"),
    "Naive Bayes": joblib.load("model/Naive_Bayes.pkl"),
    "Random Forest": joblib.load("model/Random_Forest.pkl"),
    "XGBoost": joblib.load("model/XGBoost.pkl")
}


# User Inputs
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])
model_name = st.selectbox("Select Model", models.keys())


# Main Processing
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if "Depression" not in df.columns:
        st.error("Column 'Depression' not found.")
        st.stop()

    # Split features and target
    X = df.drop("Depression", axis=1)
    y = df["Depression"]

    # Handle missing values
    for col in X.columns:
        X[col].fillna(
            X[col].mode()[0] if X[col].dtype == "object" else X[col].median(),
            inplace=True
        )

    # Encode categorical columns
    for col, encoder in label_encoders.items():
        if col in X.columns:
            X[col] = encoder.transform(X[col])

    # Scale features
    X_scaled = scaler.transform(X)

    # Predict
    model = models[model_name]
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    
    # Evaluation Metrics
    st.subheader("Evaluation Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y, y_pred):.3f}")
    col2.metric("Precision", f"{precision_score(y, y_pred):.3f}")
    col3.metric("Recall", f"{recall_score(y, y_pred):.3f}")
    col4.metric("F1 Score", f"{f1_score(y, y_pred):.3f}")

    
    # Classification Report (Readable)
    st.subheader("Classification Report")

    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    report_df.rename(
        index={"0": "Not Depressed", "1": "Depressed"},
        inplace=True
    )

    st.dataframe(report_df.round(3))

    
    # Confusion Matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Depressed", "Depressed"],
        yticklabels=["Not Depressed", "Depressed"]
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    
    # ROC Curve
    st.subheader("ROC Curve")

    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

# 🎓 Student Depression Prediction

## Project Description:

The machine learning project focuses on predicting student depression using multiple classification techniques. Six different algorithms are trained and evaluated using standard performance metrics to identify the most reliable model. The project incorporates data preprocessing, comprehensive model evaluation, and an interactive Streamlit application to enable early risk detection and support timely academic intervention.

---

## 📌 Problem Statement:

Student depression is a growing concern that negatively impacts academic performance, mental well-being, and long-term career outcomes. Early identification of students at risk can help institutions provide timely counseling and support. This project aims to build and evaluate machine learning models that can accurately predict depression among students based on survey and behavioral data.

---

## 📊 Dataset Description:

The dataset consists of student-related attributes such as demographic details, academic pressure, lifestyle factors, and mental health indicators. The target variable represents whether a student is experiencing depression. The dataset was preprocessed by handling missing values, encoding categorical variables, and scaling numerical features where required.

**Key Characteristics:**

* Binary classification problem
* Moderate class imbalance
* Combination of numerical and categorical features

---

## 🤖 Models Used & Performance Comparison:

Six machine learning models were trained and evaluated using Accuracy, AUC, Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC).

### 📈 Machine Learbing Model Performance Comparison Table:

| ML Model Name            | Accuracy | AUC   | Precision | Recall | F1    | MCC    |
| ------------------------ | -------- | ----- | --------- | ------ | ----- | ------ |
| Logistic Regression      | 0.843    | 0.917 | 0.856     | 0.879  | 0.867 | 0.674  |
| Decision Tree            | 0.765    | 0.759 | 0.803     | 0.793  | 0.798 | 0.517  |
| kNN                      | 0.813    | 0.870 | 0.826     | 0.864  | 0.844 | 0.613  |
| Naive Bayes              | 0.585    | 0.914 | 0.585     | 0.999  | 0.738 | -0.016 |
| Random Forest (Ensemble) | 0.839    | 0.914 | 0.853     | 0.875  | 0.864 | 0.666  |
| XGBoost (Ensemble)       | 0.833    | 0.910 | 0.851     | 0.867  | 0.859 | 0.654  |

---

## 🧐 Observations on Model Performance:

| ML Model Name            | Observation about model performance                                                                                                                              |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Performed strongly with high accuracy (84.3%), AUC (0.917), and MCC (0.674), indicating good linear separability and balanced predictions.                       |
| Decision Tree            | Showed moderate performance with lower accuracy (76.5%) and MCC (0.517). Prone to overfitting, which reduced generalization.                                     |
| kNN                      | Achieved good accuracy (81.3%) and F1-score (0.844). Performance benefited from feature scaling but is sensitive to noise and dataset size.                      |
| Naive Bayes              | Exhibited extremely high recall (0.999) but low accuracy (58.5%) and negative MCC (-0.016), indicating heavy bias toward the positive class.                     |
| Random Forest (Ensemble) | Delivered robust and consistent performance with high accuracy (83.9%) and MCC (0.666). Ensemble learning reduced overfitting effectively.                       |
| XGBoost (Ensemble)       | Provided consistently high performance with strong accuracy (83.3%), F1-score (0.859), and MCC (0.654). Gradient boosting captured complex patterns efficiently. |

---

## 🏆 Key Takeaways:

* **Best Overall Models:** Logistic Regression, Random Forest, and XGBoost
* **Highest Recall:** Naive Bayes (but at the cost of precision)
* **Most Balanced Model:** Logistic Regression (highest MCC)
* Ensemble models generally outperform single estimators

---

## 🛠️ Tech Stack:

* Python 3.9+
* NumPy, Pandas
* Scikit-learn
* XGBoost
* Matplotlib / Seaborn
* Streamlit (for deployment)

---

## 🚀 How to Run:

```bash
pip3 install -r requirements.txt
python3 train_models.py
python3 -m streamlit run app.py
```

---

## 📌 Conclusion:

This project demonstrates how machine learning can be effectively used for early detection of student depression. The comparative analysis highlights that ensemble and linear models provide the most reliable predictions, making them suitable candidates for real-world academic support systems.

---


Telco Customer Churn Prediction 📖 Overview

This project builds a Machine Learning classification model to predict whether a telecom customer will churn based on demographic details, service subscriptions, and billing information.

The objective is to help telecom companies identify high-risk customers and improve retention strategies.

🎯 Problem Statement

Customer churn significantly impacts business revenue. Since the dataset is imbalanced, relying only on accuracy can be misleading. The goal was to build a robust classification model while properly handling class imbalance and optimizing meaningful evaluation metrics.

⚙️ Methodology

Performed data preprocessing and categorical encoding

Handled class imbalance using SMOTE

Applied Cross-Validation

Tuned hyperparameters using GridSearchCV

Optimized model using Macro F1-score instead of accuracy

Model Used

Random Forest Classifier

📊 Model Performance Metric Score Accuracy 77% Macro F1-score 0.73 Recall (Churn Class) 0.69 Confusion Matrix [[833 203] [116 257]]

The model improves minority class detection (churn customers) while maintaining balanced overall performance.

🛠 Tech Stack

Python

Pandas

NumPy

Scikit-learn

imbalanced-learn (SMOTE)

pickle

▶️ How to Run 1️⃣ Install Dependencies pip install -r requirements.txt

2️⃣ Load Model and Predict import pickle import pandas as pd

model = pickle.load("model.pkl")

data = {...} # input dictionary df = pd.DataFrame([data])

prediction = model.predict(df) print(prediction)

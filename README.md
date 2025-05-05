# 💳 Credit Card Fraud Detection System

This project implements a machine learning system to detect fraudulent credit card transactions using the popular creditcard.csv dataset. The system is trained on real transaction data and uses advanced techniques like data scaling, oversampling (SMOTE), and a Random Forest classifier to identify fraud with high accuracy.

---

## 📌 Problem Statement

Credit card fraud poses a major threat to online financial systems. The goal of this project is to develop an AI-based model that can accurately detect fraudulent transactions in real time, minimizing financial losses and increasing security.

---

## 🎯 Objective

- Build a fraud detection model using machine learning.
- Improve fraud detection accuracy by handling data imbalance.
- Save the trained model for future real-time deployment.

---

## 📊 Dataset

- *Source*: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- *Records*: 284,807 transactions
- *Fraud Cases*: 492 (~0.17%)
- *Attributes*: Time, Amount, V1-V28 (anonymized features), Class (target)

---

## ⚙ Tools & Technologies

- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Joblib
- Jupyter Notebook

---

## 🧠 Model Used

- *Random Forest Classifier*
- *SMOTE (Synthetic Minority Oversampling Technique)* to balance class distribution

---

## 📁 Files

- creditcard.csv – Transaction dataset
- fraud_detection.py – Main program
- fraud_model.pkl – Saved model
- scaler.pkl – Saved scaler

---

## 🏁 How to Run

1. Install dependencies:
   ```bash
   pip install pandas scikit-learn imbalanced-learn joblib

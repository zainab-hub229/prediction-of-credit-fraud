

# Credit Card Fraud Detection using Machine Learning

## Name: Zainab

## Project Overview
This project is based on detecting fraudulent credit card transactions using Machine Learning.
The main challenge is that fraud transactions are very rare compared to normal transactions, so the project focuses on handling imbalanced data and evaluating models using Precision, Recall, and F1-score.

## Dataset
Dataset: Credit Card Fraud Detection (Kaggle)

Dataset contains:
- 284,807 total transactions
- 492 fraud transactions
- 284,315 normal transactions

Features:
- Time
- Amount
- V1 to V28

Target:
- Class
  - 0 = Normal Transaction
  - 1 = Fraud Transaction

## Exploratory Data Analysis (EDA)
Performed:
- Checked dataset shape
- Checked columns and data types
- Checked missing values
- Visualized fraud vs normal transaction distribution

## Data Preprocessing
Applied:
- StandardScaler on Amount and Time columns
- Train-test split
- Stratified splitting to maintain class ratio

## Handling Imbalanced Data
Used:
- class_weight='balanced'

This helps the model learn from rare fraud transactions.

## Machine Learning Models

Implemented models:

1. Logistic Regression
2. Random Forest Classifier

## Evaluation Metrics

Models were evaluated using:

- Confusion Matrix
- Precision
- Recall
- F1-score
- ROC-AUC Curve

## Conclusion

Random Forest performed better because it provided better fraud detection performance.
Recall and F1-score were considered important because identifying fraud transactions is the main objective.

## Project Files

- analysis.py → Data analysis and visualization
- model.py → Machine learning model training
- app.py → Streamlit frontend application
- creditcard.csv → Dataset

## Github Repository

https://github.com/zainab-hub229/Credit-fraud-prediction.git

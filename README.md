# Credit Card Fraud Detection using Machine Learning

## Name: Zainab

## Project Overview
This project focuses on detecting fraudulent credit card transactions using Machine Learning. 
The main challenge is handling imbalanced data because fraud transactions are very rare compared to normal transactions.

## Dataset
Credit Card Fraud Detection Dataset (Kaggle)

- Total Transactions: 284,807
- Fraud Cases: 492
- Normal Cases: 284,315

Features:
- Time
- Amount
- V1 - V28

Target:
- Class
  - 0 = Normal
  - 1 = Fraud

## Steps Performed

### Exploratory Data Analysis (EDA)
- Checked dataset shape
- Checked columns and data types
- Checked missing values
- Visualized fraud vs normal transactions

### Preprocessing
- Applied StandardScaler on Amount and Time
- Performed train-test split
- Used stratified splitting

### Handling Imbalanced Data
Used class_weight='balanced' to handle fraud and normal transaction imbalance.

### Machine Learning Models
Implemented:

1. Logistic Regression
2. Random Forest Classifier

## Evaluation Metrics

Models were evaluated using:

- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC Curve

Accuracy was not considered the main metric because the dataset is highly imbalanced.

## Conclusion

Random Forest performed better because it was able to identify fraud transactions more effectively. 
Recall and F1-score were considered important metrics because detecting fraud cases is the main objective.

## Project Files

- analysis.py → EDA and visualization
- model.py → Model training and evaluation
- app.py → Streamlit frontend
- creditcard.csv → Dataset



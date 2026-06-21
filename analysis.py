# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset
df = pd.read_csv("creditcard.csv")


# Check shape
print(df.shape)


# Columns
print(df.columns)


# Data info
print(df.info())


# Missing values
print(df.isnull().sum())


# Fraud count
print(df["Class"].value_counts())


# Class distribution graph
sns.countplot(x="Class", data=df)

plt.title("Fraud vs Normal Transactions")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Count")

plt.show()


# Statistics
print(df[["Amount", "Time"]].describe())

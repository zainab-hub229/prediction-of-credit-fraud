# Import libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load dataset
df = pd.read_csv("creditcard.csv")


# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]


# Scale Amount and Time
scaler = StandardScaler()

X["Amount"] = scaler.fit_transform(X[["Amount"]])
X["Time"] = scaler.fit_transform(X[["Time"]])


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Check data
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# Check fraud ratio
print("Training class:")
print(y_train.value_counts())

print("Testing class:")
print(y_test.value_counts())

# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ==========================
# Logistic Regression Model
# ==========================

lr_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

# Train model
lr_model.fit(X_train, y_train)


# Prediction
lr_prediction = lr_model.predict(X_test)


# Evaluation
print("Logistic Regression Results")

print("Accuracy:",
      accuracy_score(y_test, lr_prediction))

print(classification_report(
    y_test,
    lr_prediction
))


# ==========================
# Random Forest Model
# ==========================

rf_model = RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)


# Train
rf_model.fit(X_train, y_train)


# Prediction
rf_prediction = rf_model.predict(X_test)


# Evaluation
print("Random Forest Results")

print("Accuracy:",
      accuracy_score(y_test, rf_prediction))

print(classification_report(
    y_test,
    rf_prediction
))


# Confusion Matrix

print("Confusion Matrix Random Forest")

print(confusion_matrix(
    y_test,
    rf_prediction
))
# Import visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc


# ==========================
# Confusion Matrix Heatmap
# ==========================

cm = confusion_matrix(
    y_test,
    rf_prediction
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()



# ==========================
# ROC-AUC Curve
# ==========================

rf_probability = rf_model.predict_proba(X_test)[:,1]


fpr, tpr, threshold = roc_curve(
    y_test,
    rf_probability
)


roc_auc = auc(
    fpr,
    tpr
)


plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.2f}"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC-AUC Curve")

plt.legend()

plt.show()

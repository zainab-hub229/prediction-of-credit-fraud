import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection System")
st.write("Saari 30 features daalo, model batayega Fraud hai ya Normal")

@st.cache_data
def load_data():
    df = pd.read_csv('creditcard.csv')
    return df

@st.cache_resource
def train_model(df):
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model, X.columns

df = load_data()
model, feature_columns = train_model(df)

st.subheader("Transaction Details Daalo:")
col1, col2, col3, col4, col5, col6 = st.columns(6)

inputs = []
for i, col_name in enumerate(feature_columns):
    col = [col1, col2, col3, col4, col5, col6][i % 6]
    with col:
        val = st.number_input(f"{col_name}", value=0.0, format="%.6f", key=col_name)
        inputs.append(val)

if st.button("🔍 Predict Karo", type="primary"):
    input_array = np.array(inputs).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    prob = model.predict_proba(input_array)[0]

    if prediction == 1:
        st.error(f"🚨 ALERT: Fraud Transaction Detected! \nProbability: {prob[1]*100:.2f}%")
    else:
        st.success(f"✅ Safe Transaction \nProbability: {prob[0]*100:.2f}%")

st.info("💡 Tip: Agar saari values 0.0 rakh kar Predict dabao ge to 'Safe Transaction' aayega.")

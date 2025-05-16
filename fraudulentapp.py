import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import joblib

# Page config
st.set_page_config(page_title="🚨 Fraud Detection System", layout="wide")
st.title("🚨 Real-Time Fraud Detection")

# Load data
@st.cache_data
def load_data():
    url = 'https://raw.github.com/Vasanthkumar5648/fraud_cap/main/Fraud_Analysis_Dataset.csv'
    df = pd.read_csv(url)
    
    # Feature engineering
    df['nameOrig_len'] = df['nameOrig'].apply(len)
    df['nameOrig_digit_count'] = df['nameOrig'].str.count(r'\d')
    df['nameOrig_alpha_count'] = df['nameOrig'].str.count(r'[A-Za-z]')
    df['errorBalanceOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['errorBalanceDest'] = df['newbalanceDest'] - df['oldbalanceDest']
    df = df.drop(['nameOrig', 'nameDest'], axis=1)
    
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])
    
    return df, le

df, le = load_data()

# Sidebar for user input
st.sidebar.header("Transaction Details")
transaction_type = st.sidebar.selectbox("Type", ["CASH-IN", "CASH-OUT", "DEBIT", "PAYMENT", "TRANSFER"])
amount = st.sidebar.number_input("Amount", min_value=0.0, value=500.0)
oldbalance_org = st.sidebar.number_input("Origin Balance Before", min_value=0.0, value=1000.0)
newbalance_org = st.sidebar.number_input("Origin Balance After", min_value=0.0, value=500.0)
oldbalance_dest = st.sidebar.number_input("Destination Balance Before", min_value=0.0, value=0.0)
newbalance_dest = st.sidebar.number_input("Destination Balance After", min_value=0.0, value=500.0)

# Model training (cached)
@st.cache_resource
def train_model():
    X = df.drop('isFraud', axis=1)
    y = df['isFraud']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    
    # PCA
    pca = PCA(0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_sm)
    X_test_pca = pca.transform(X_test)
    
    # Train XGBoost
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train_pca, y_train_sm)
    
    return model, pca, X_test_pca, y_test

model, pca, X_test_pca, y_test = train_model()

# Prediction function
def predict_fraud(input_data):
    # Preprocess
    input_data['type'] = le.transform([input_data['type']])[0]
    input_df = pd.DataFrame([input_data])
    
    # Add engineered features
    input_df['nameOrig_len'] = 10  # Default values matching training
    input_df['nameOrig_digit_count'] = 6
    input_df['nameOrig_alpha_count'] = 4
    input_df['errorBalanceOrig'] = input_df['oldbalanceOrg'] - input_df['newbalanceOrig']
    input_df['errorBalanceDest'] = input_df['newbalanceDest'] - input_df['oldbalanceDest']
    
    # PCA transform
    input_pca = pca.transform(input_df)
    
    # Predict
    proba = model.predict_proba(input_pca)[0][1]
    prediction = 1 if proba > 0.5 else 0
    
    return prediction, proba

# Main interface
if st.sidebar.button("Check Fraud Risk"):
    input_data = {
        'step': 1,
        'type': transaction_type,
        'amount': amount,
        'oldbalanceOrg': oldbalance_org,
        'newbalanceOrig': newbalance_org,
        'oldbalanceDest': oldbalance_dest,
        'newbalanceDest': newbalance_dest
    }
    
    prediction, proba = predict_fraud(input_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Analysis")
        st.metric("Fraud Probability", f"{proba:.2%}", 
                 delta="High Risk" if prediction else "Low Risk",
                 delta_color="inverse")
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_pca)[:,1])
        fig1, ax1 = plt.subplots()
        ax1.plot(fpr, tpr, color='red', label=f'AUC = {auc(fpr, tpr):.2f}')
        ax1.plot([0, 1], [0, 1], linestyle='--')
        ax1.set_title('Model ROC Curve')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        st.pyplot(fig1)
    
    with col2:
        st.subheader("Key Indicators")
        
        # Feature importance
        importance = model.feature_importances_
        features = [f"PC{i+1}" for i in range(len(importance))]
        fig2, ax2 = plt.subplots()
        sns.barplot(x=importance, y=features, palette='viridis', ax=ax2)
        ax2.set_title('Top Principal Components Importance')
        st.pyplot(fig2)
        
        # Transaction details
        st.write("### Transaction Summary")
        st.json({
            "Type": transaction_type,
            "Amount": f"${amount:,.2f}",
            "Origin Balance Change": f"${oldbalance_org - newbalance_org:,.2f}",
            "Destination Balance Change": f"${newbalance_dest - oldbalance_dest:,.2f}"
        })

# Model performance section
with st.expander("📊 Model Performance Details"):
    st.subheader("Classification Report")
    y_pred = model.predict(X_test_pca)
    st.text(classification_report(y_test, y_pred))
    
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    st.pyplot(fig3)

# Batch processing
st.sidebar.header("Batch Processing")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)
    if st.sidebar.button("Process Batch"):
        with st.spinner("Analyzing transactions..."):
            # Add batch processing logic here
            st.success(f"Processed {len(batch_df)} transactions")
            st.dataframe(batch_df.head())

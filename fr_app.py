import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# Title and description
st.title("💳 Fraud Detection System")
st.markdown("""
This application detects potentially fraudulent financial transactions using machine learning.
""")

# Load data function with caching
@st.cache_data
def load_and_preprocess_data():
    # Load data
    df = pd.read_csv('https://raw.github.com/Vasanthkumar5648/fraud_cap/main/Fraud_Analysis_Dataset.csv')
    
    # Preprocessing
    df['type'] = LabelEncoder().fit_transform(df['type'])
    df['errorBalanceOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['errorBalanceDest'] = df['newbalanceDest'] - df['oldbalanceDest']
    df = df.drop(['nameOrig', 'nameDest'], axis=1)
    
    # Split data
    X = df.drop('isFraud', axis=1)
    y = df['isFraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    
    # Apply PCA
    pca = PCA(0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_sm)
    X_test_pca = pca.transform(X_test)
    
    # Train model
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train_pca, y_train_sm)
  
  # Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home","Transaction Checker"])

if page == "Home":
    st.header("About This Application")
    st.write("""
    This fraud detection system helps identify suspicious financial transactions using advanced machine learning techniques.
    
    ### Key Features:
    - Uses XGBoost classifier for high accuracy
    - Handles class imbalance with SMOTE
    - Reduces dimensionality with PCA
    - Provides detailed performance metrics
    - Interactive transaction checking
    
    ### Dataset Information:
    The model was trained on a dataset containing:
    - Over 6 million transactions
    - 10 features per transaction
    - Highly imbalanced classes (fraud cases are rare)
    """)
    
elif page == "Transaction Checker":
    st.header("Transaction Fraud Checker")
    st.write("Enter transaction details to check for potential fraud:")
    
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            step = st.selectbox("Hour of Transaction (1-744)",list(range(1,744,10)))
            transaction_type = st.selectbox("Transaction Type", 
                                          ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
            amount = st.selectbox("Amount ($)", list(range(0,1000,10)))
            
        with col2:
            oldbalanceOrg = st.selectbox("Originator Old Balance", 
                                        list(range(0,1000,10)))
            newbalanceOrig = st.selectbox("Originator New Balance", 
                                           list(range(0,1000,10)))
            oldbalanceDest = st.selectbox("Destination Old Balance", 
                                           list(range(0,1000,10)))
            newbalanceDest = st.selectbox("Destination New Balance", 
                                           list(range(0,1000,10)))
        
        submitted = st.form_submit_button("Check Transaction")
    if submitted:
        # Process input
        type_mapping = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}
        transaction_type_encoded = type_mapping[transaction_type]
        
        errorBalanceOrig = oldbalanceOrg - newbalanceOrig
        errorBalanceDest = newbalanceDest - oldbalanceDest
        
        features = np.array([[step, transaction_type_encoded, amount, 
                            oldbalanceOrg, newbalanceOrig, 
                            oldbalanceDest, newbalanceDest,
                            errorBalanceOrig, errorBalanceDest]])
        
        # Apply PCA and predict
        def load_and_preprocess_data():
        # Load data
        df = pd.read_csv('https://raw.github.com/Vasanthkumar5648/fraud_cap/main/Fraud_Analysis_Dataset.csv')
    
        # Preprocessing
        df['type'] = LabelEncoder().fit_transform(df['type'])
        df['errorBalanceOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
        df['errorBalanceDest'] = df['newbalanceDest'] - df['oldbalanceDest']
        df = df.drop(['nameOrig', 'nameDest'], axis=1)
        # Split data
        X = df.drop('isFraud', axis=1)
        y = df['isFraud']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
        # Handle imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
        from sklearn.decomposition import PCA
    
        pca = PCA(0.95, random_state=42)
        X_train_pca = pca.fit_transform(X_train_sm)
        X_test_pca = pca.transform(X_test)
        features_pca = pca.transform(features)
        prediction = model.predict(features_pca)
        prediction_proba = model.predict_proba(features_pca)

        # Display results
        st.subheader("Result")
        if prediction[0] == 1:
            st.error("🚨 Fraud Detected!")
            st.warning("This transaction has been flagged as potentially fraudulent.")
        else:
            st.success("✅ Legitimate Transaction")
            st.info("This transaction appears to be legitimate.")
        
        st.write(f"Fraud Probability: {prediction_proba[0][1]*100:.2f}%")
        
        # Show probability gauge
        fig, ax = plt.subplots(figsize=(6, 1))
        ax.barh(['Fraud Risk'], [prediction_proba[0][1]], color='red' if prediction[0] == 1 else 'green')
        ax.set_xlim(0, 1)
        ax.set_title('Fraud Probability Gauge')
        st.pyplot(fig)

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
page = st.sidebar.radio("Go to", ["Home", "Model Performance", "Transaction Checker"])

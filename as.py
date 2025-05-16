import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, classification_report, auc
from xgboost import XGBClassifier
from sklearn.decomposition import PCA

df =pd.read_csv('https://raw.github.com/Vasanthkumar5648/fraud_cap/main/Fraud_Analysis_Dataset.csv')df['type'] = LabelEncoder().fit_transform(df['type'])

df = df.drop(['nameOrig', 'nameDest'], axis=1)

# Feature: Difference in balances
df['errorBalanceOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['errorBalanceDest'] = df['newbalanceDest'] - df['oldbalanceDest']

df = df.drop(['nameOrig', 'nameDest'], axis=1)

# Define features and target
X = df_model.drop('isFraud', axis=1)
y = df_model['isFraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

import streamlit as st
st.title("💳 Fraud Detection System")


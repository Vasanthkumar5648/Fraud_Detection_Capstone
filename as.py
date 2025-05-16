import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, classification_report, auc
from xgboost import XGBClassifier
from sklearn.decomposition import PCA

df =pd.read_csv('https://raw.github.com/Vasanthkumar5648/fraud_cap/main/Fraud_Analysis_Dataset.csv')

df['type'] = LabelEncoder().fit_transform(df['type'])

# Feature: Difference in balances
df['errorBalanceOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['errorBalanceDest'] = df['newbalanceDest'] - df['oldbalanceDest']

df = df.drop(['nameOrig', 'nameDest'], axis=1)

# Define features and target
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

from sklearn.decomposition import PCA
# PCA for dimensionality reduction (keep 95% variance)
pca = PCA(0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_sm)
X_test_pca = pca.transform(X_test)

models = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train_pca, y_train_sm)
import streamlit as st
st.title("💳 Fraud Detection System")


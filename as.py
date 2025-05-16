import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, classification_report, auc
from xgboost import XGBClassifier
from sklearn.decomposition import PCA

df =pd.read_csv('https://raw.github.com/Vasanthkumar5648/fraud_cap/main/Fraud_Analysis_Dataset.csv')

# Preprocessing
df_model = df.copy()

# Extract basic NLP features from nameOrig
df_model['nameOrig_len'] = df_model['nameOrig'].apply(len)
df_model['nameOrig_digit_count'] = df_model['nameOrig'].str.count(r'\d')
df_model['nameOrig_alpha_count'] = df_model['nameOrig'].str.count(r'[A-Za-z]')

# Drop high-cardinality columns
df_model.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)

# Encode 'type'
le = LabelEncoder()
df_model['type'] = le.fit_transform(df_model['type'])

# Define features and target
X = df_model.drop('isFraud', axis=1)
y = df_model['isFraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from imblearn.over_sampling import SMOTE

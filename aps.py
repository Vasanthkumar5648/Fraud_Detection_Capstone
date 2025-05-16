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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve,roc_auc_score,accuracy_score,auc


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
    
    return model, pca, X_test_pca, y_test

# Load data and model
model, pca, X_test_pca, y_test = load_and_preprocess_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Model Performance", "Transaction Checker"])

# Home page
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
    
    st.image("https://miro.medium.com/v2/resize:fit:1400/1*_91k5Q9wZ5x3oZxJhxHZPQ.png", 
             caption="Fraud Detection Concept", width=600)

# Model Performance page
elif page == "Model Performance":
    st.header("Model Performance Metrics")
    
    # Make predictions
    y_pred = model.predict(X_test_pca)
    y_pred_proba = model.predict_proba(X_test_pca)[:, 1]
    
    # Classification report
    st.subheader("Classification Report")
    st.code(classification_report(y_test, y_pred))
    
    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Fraud'], 
                yticklabels=['Legitimate', 'Fraud'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    
    # ROC curve
    st.subheader("ROC Curve")
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)
    
    # Metrics
    st.metric("ROC AUC Score", f"{roc_auc_score(y_test, y_pred_proba):.4f}")
    st.metric("Accuracy Score", f"{accuracy_score(y_test, y_pred):.4f}")
    st.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
    st.metric("Recall",f"{recall_score(y_test, y_pred):.4f}")
    st.metric("F1 Score",f"{f1_score(y_test, y_pred):.4f}")

# Transaction Checker page
elif page == "Transaction Checker":
    st.header("Transaction Fraud Checker")
    st.write("Enter transaction details to check for potential fraud:")
    
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            step = st.number_input("Hour of Transaction (1-744)", 
                                  min_value=1, max_value=744, value=1)
            transaction_type = st.selectbox("Transaction Type", 
                                          ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
            amount = st.number_input("Amount ($)", min_value=0.0, value=1000.0, step=0.01)
            
        with col2:
            oldbalanceOrg = st.number_input("Originator Old Balance", 
                                          min_value=0.0, value=1000.0, step=0.01)
            newbalanceOrig = st.number_input("Originator New Balance", 
                                           min_value=0.0, value=0.0, step=0.01)
            oldbalanceDest = st.number_input("Destination Old Balance", 
                                           min_value=0.0, value=0.0, step=0.01)
            newbalanceDest = st.number_input("Destination New Balance", 
                                           min_value=0.0, value=1000.0, step=0.01)
        
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

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "Fraud Detection System v1.0\n\n"
    "For demonstration purposes only"
)

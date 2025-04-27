
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Load saved models ---
model = joblib.load('models/random_forest.pkl')  # You can switch to any model you prefer!

# --- Load feature list ---
FEATURES = ['amount', 'hour', 'is_high_amount', 'is_type_transfer_or_cashout', 
            'error_balance_orig', 'error_balance_dest', 'type_encoded']

# --- App Title ---
st.title("💳 Fraud Detection System")

st.markdown("""
Welcome to the Fraud Detection App!  
Upload your transaction CSV file or input details manually to predict fraudulent transactions.  
**Model:** Random Forest (pre-trained)
""")

# --- Sidebar for mode selection ---
mode = st.sidebar.selectbox("Choose Input Method:", ["Upload CSV File", "Manual Entry"])

def preprocess_data(df):
    # Ensure the dataframe has necessary features
    missing_cols = set(FEATURES) - set(df.columns)
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
        return None
    return df[FEATURES]

# --- Prediction ---
def predict(data):
    preds = model.predict(data)
    preds_proba = model.predict_proba(data)[:, 1]
    return preds, preds_proba

# --- CSV Upload ---
if mode == "Upload CSV File":
    uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        processed_df = preprocess_data(df)
        if processed_df is not None:
            st.write("✅ Data Preview:")
            st.dataframe(processed_df.head())

            preds, preds_proba = predict(processed_df)

            df['Fraud Prediction'] = preds
            df['Fraud Probability'] = preds_proba

            st.write("🔍 Prediction Results:")
            st.dataframe(df[['Fraud Prediction', 'Fraud Probability']])

            st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv")

# --- Manual Entry ---
else:
    st.subheader("Manual Transaction Entry")

    amount = st.number_input("Transaction Amount:", min_value=0.0, value=1000.0)
    hour = st.slider("Transaction Hour (0-23):", 0, 23, 12)
    is_high_amount = st.selectbox("Is High Amount (> Threshold)?", [0, 1])
    is_type_transfer_or_cashout = st.selectbox("Is TRANSFER or CASH_OUT type?", [0, 1])
    error_balance_orig = st.number_input("Error Balance Orig:", min_value=-10000.0, max_value=10000.0, value=0.0)
    error_balance_dest = st.number_input("Error Balance Dest:", min_value=-10000.0, max_value=10000.0, value=0.0)
    type_encoded = st.selectbox("Type Encoded (TRANSFER=1, CASH_OUT=2, Others=0):", [0, 1, 2])

    input_dict = {
        'amount': amount,
        'hour': hour,
        'is_high_amount': is_high_amount,
        'is_type_transfer_or_cashout': is_type_transfer_or_cashout,
        'error_balance_orig': error_balance_orig,
        'error_balance_dest': error_balance_dest,
        'type_encoded': type_encoded
    }

    input_df = pd.DataFrame([input_dict])

    if st.button("Predict Transaction Fraud"):
        preds, preds_proba = predict(input_df)

        if preds[0] == 1:
            st.error(f"🚨 Fraudulent Transaction Detected! (Probability: {preds_proba[0]:.2f})")
        else:
            st.success(f"✅ Legitimate Transaction. (Probability: {preds_proba[0]:.2f})")

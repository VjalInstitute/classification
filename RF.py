import streamlit as st
import pandas as pd
import pickle

# App Title
st.title("Random Forest Prediction App with Alphanumeric Input")

# Load Pre-trained Model
st.sidebar.header("Model Selection")
uploaded_model = st.sidebar.file_uploader("Upload your Random Forest model (.pkl)", type=["pkl"])

# Load Encoders
uploaded_encoders = st.sidebar.file_uploader("Upload your Label Encoders (.pkl)", type=["pkl"])

if uploaded_model is not None and uploaded_encoders is not None:
    rf_model = pickle.load(uploaded_model)
    label_encoders = pickle.load(uploaded_encoders)  # This should contain encoders for all categorical columns
    
    st.write("Model and encoders loaded successfully! You can now make predictions.")
    
    # Prediction Section
    st.write("### Provide Input Features for Prediction")
    
    # Define the input fields based on the original alphanumeric input
    COMPANY_NAME = st.text_input("COMPANY_NAME (e.g., ABC Corp)")
    BANK_TRANSACTION_REFERENCE = st.text_input("BANK_TRANSACTION_REFERENCE (e.g., Ref12345)")
    BANK_TRANSACTION_TYPE = st.text_input("BANK_TRANSACTION_TYPE (e.g., Credit)")
    BANK_TRANSACTION_TOTAL_AMOUNT = st.number_input("BANK_TRANSACTION_TOTAL_AMOUNT (e.g., 12345.67)", value=0.0)
    LINE_ITEM_DESCRIPTION = st.text_input("LINE_ITEM_DESCRIPTION (e.g., Payment for Invoice)")
    COMPANY_ID = st.number_input("COMPANY_ID", value=0)

    if st.button("Predict"):
        # Preprocess the inputs using the loaded encoders
        input_data = {
            "COMPANY_ID": COMPANY_ID,
            "COMPANY_NAME": label_encoders["COMPANY_NAME"].transform([COMPANY_NAME])[0],
            "BANK_TRANSACTION_REFERENCE": label_encoders["BANK_TRANSACTION_REFERENCE"].transform([BANK_TRANSACTION_REFERENCE])[0],
            "BANK_TRANSACTION_TYPE": label_encoders["BANK_TRANSACTION_TYPE"].transform([BANK_TRANSACTION_TYPE])[0],
            "BANK_TRANSACTION_TOTAL_AMOUNT": BANK_TRANSACTION_TOTAL_AMOUNT,
            "LINE_ITEM_DESCRIPTION": label_encoders["LINE_ITEM_DESCRIPTION"].transform([LINE_ITEM_DESCRIPTION])[0],
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Predict
        prediction = rf_model.predict(input_df)[0]

        # Reverse the encoded prediction to its original value (if necessary)
        predicted_account_name = label_encoders["RECONCILITION_ACCOUNT_NAME"].inverse_transform([prediction])[0]

        st.write(f"### Predicted Reconciliation Account Name: {predicted_account_name}")
else:
    st.write("Upload both a trained Random Forest model and corresponding label encoders to get started.")


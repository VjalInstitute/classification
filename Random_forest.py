import streamlit as st
import pandas as pd
import pickle

# App Title
st.title("Random Forest Prediction App with Pre-trained Model")

# Load the Pre-trained Model
@st.cache_resource
def load_model():
    try:
        with open("rf_model.pkl", "rb") as model_file:
            rf_model = pickle.load(model_file)
        return rf_model
    except FileNotFoundError:
        st.error("The pre-trained model (rf_model.pkl) is missing. Please ensure it is available in the same directory as the app.")
        return None

# Load the model
rf_model = load_model()

# Upload Encoders
uploaded_encoders = st.sidebar.file_uploader("Upload your Label Encoders (.pkl)", type=["pkl"])

if rf_model and uploaded_encoders is not None:
    try:
        # Load the label encoders
        label_encoders = pickle.load(uploaded_encoders)
        st.success("Model and encoders loaded successfully! You can now make predictions.")

        # Prediction Section
        st.write("### Provide Input Features for Prediction")

        # Input Fields for Alphanumeric Data
        COMPANY_NAME = st.text_input("COMPANY_NAME (e.g., ABC Corp)")
        BANK_TRANSACTION_REFERENCE = st.text_input("BANK_TRANSACTION_REFERENCE (e.g., Ref12345)")
        BANK_TRANSACTION_TYPE = st.text_input("BANK_TRANSACTION_TYPE (e.g., Credit)")
        BANK_TRANSACTION_TOTAL_AMOUNT = st.number_input("BANK_TRANSACTION_TOTAL_AMOUNT (e.g., 12345.67)", value=0.0)
        LINE_ITEM_DESCRIPTION = st.text_input("LINE_ITEM_DESCRIPTION (e.g., Payment for Invoice)")
        COMPANY_ID = st.number_input("COMPANY_ID (e.g., 1234)", value=0)

        if st.button("Predict"):
            try:
                # Preprocess the inputs using the uploaded label encoders
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

                # Decode the predicted class
                predicted_account_name = label_encoders["RECONCILITION_ACCOUNT_NAME"].inverse_transform([prediction])[0]

                st.write(f"### Predicted Reconciliation Account Name: {predicted_account_name}")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    except Exception as e:
        st.error(f"Error loading encoders: {str(e)}")
else:
    if not rf_model:
        st.warning("The pre-trained model is not available. Please ensure 'rf_model.pkl' is in the same directory.")
    if not uploaded_encoders:
        st.warning("Please upload a valid Label Encoder file to get started.")

import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# App title
st.title("Random Forest Prediction App")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    try:
        # Load and preview the dataset
        chunk_size = 5000
        columns_needed = [
            'COMPANY_ID', 'COMPANY_NAME', 'BANK_TRANSACTION_REFERENCE',
            'BANK_TRANSACTION_TYPE', 'BANK_TRANSACTION_TOTAL_AMOUNT',
            'LINE_ITEM_DESCRIPTION', 'RECONCILITION_ACCOUNT_NAME'
        ]

        processed_chunks = []
        uploaded_file.seek(0)
        for chunk in pd.read_csv(uploaded_file, chunksize=chunk_size, usecols=columns_needed, low_memory=False):
            chunk.fillna('Unknown', inplace=True)
            processed_chunks.append(chunk)

        data = pd.concat(processed_chunks, ignore_index=True)

        st.write("### Dataset Preview")
        st.dataframe(data.head())

        # Separate numeric and categorical columns
        numeric_columns = ['BANK_TRANSACTION_TOTAL_AMOUNT']
        categorical_columns = ['BANK_TRANSACTION_REFERENCE', 'LINE_ITEM_DESCRIPTION',
                               'COMPANY_NAME', 'BANK_TRANSACTION_TYPE',
                               'RECONCILITION_ACCOUNT_NAME']

        # Fill numeric columns with a default numeric value (e.g., 0.0)
        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')  # Ensure values are numeric
            data[col] = data[col].fillna(0.0)  # Replace NaN with 0.0

        # Ensure categorical columns are strings
        data[categorical_columns] = data[categorical_columns].astype('str')

        # Encode categorical columns
        label_encoders = {}
        for column in categorical_columns:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            label_encoders[column] = le

        # Prepare features and target
        X = data[[
            'COMPANY_ID', 'COMPANY_NAME', 'BANK_TRANSACTION_REFERENCE',
            'BANK_TRANSACTION_TYPE', 'BANK_TRANSACTION_TOTAL_AMOUNT',
            'LINE_ITEM_DESCRIPTION'
        ]]
        y = data['RECONCILITION_ACCOUNT_NAME']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train the Random Forest model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Evaluate the model
        rf_y_pred = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_y_pred)
        st.write("### Model Accuracy")
        st.write(f"Random Forest Accuracy: {rf_accuracy:.4f}")

        # Prediction section
        st.write("### Make a Prediction")

        # Input fields for prediction
        COMPANY_ID = st.number_input("COMPANY_ID", value=0)
        COMPANY_NAME = st.text_input("COMPANY_NAME", value="")
        BANK_TRANSACTION_REFERENCE = st.text_input("BANK_TRANSACTION_REFERENCE", value="Unknown")
        BANK_TRANSACTION_TYPE = st.text_input("BANK_TRANSACTION_TYPE", value="")
        BANK_TRANSACTION_TOTAL_AMOUNT = st.number_input("BANK_TRANSACTION_TOTAL_AMOUNT", value=0.0)
        LINE_ITEM_DESCRIPTION = st.text_input("LINE_ITEM_DESCRIPTION", value="Unknown")

        if st.button("Predict"):
            try:
                # Convert user inputs into encoded values
                input_data = [[
                    COMPANY_ID,
                    label_encoders['COMPANY_NAME'].transform([COMPANY_NAME])[0],
                    label_encoders['BANK_TRANSACTION_REFERENCE'].transform([BANK_TRANSACTION_REFERENCE])[0],
                    label_encoders['BANK_TRANSACTION_TYPE'].transform([BANK_TRANSACTION_TYPE])[0],
                    BANK_TRANSACTION_TOTAL_AMOUNT,  # Already numeric
                    label_encoders['LINE_ITEM_DESCRIPTION'].transform([LINE_ITEM_DESCRIPTION])[0]
                ]]

                # Make prediction
                prediction = rf_model.predict(input_data)

                # Decode the predicted value
                predicted_value = label_encoders['RECONCILITION_ACCOUNT_NAME'].inverse_transform(prediction)[0]

                st.write(f"Predicted RECONCILITION_ACCOUNT_NAME: {predicted_value}")

            except Exception as e:
                st.write("Error in prediction. Please check your inputs.")
                st.write(str(e))

    except Exception as e:
        st.write("Error processing the file. Please check the file format or content.")
        st.write(str(e))

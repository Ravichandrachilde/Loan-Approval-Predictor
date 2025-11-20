import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from streamlit_shap import st_shap  # for shap plots

# Loading model from HF repo 
@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id="Ravichandrachilde/loan-prediction-XGB",
            filename="loan_model.joblib"
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

# UI
st.title('Loan Approval Prediction')
st.info("Adjust the details below to check loan eligibility.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox('Gender', ['Male', 'Female'])
    married = st.selectbox('Married', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
    education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
    credit_history = st.selectbox('Credit History', [1.0, 0.0]) # Float to match typical datasets

with col2:
    applicant_income = st.number_input('Applicant Income', min_value=0, value=5000)
    coapplicant_income = st.number_input('Coapplicant Income', min_value=0, value=0)
    loan_amount = st.number_input('Loan Amount', min_value=0, value=100)
    loan_amount_term = st.number_input('Loan Amount Term (months)', min_value=0, value=360)
    property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])

input_data = pd.DataFrame({
    'Gender': [gender],
    'Married': [married],
    'Dependents': [dependents],
    'Education': [education],
    'Self_Employed': [self_employed],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_amount_term],
    'Credit_History': [credit_history],
    'Property_Area': [property_area]
})

# Predict Button
if st.button('Predict'):
    if model:
        with st.spinner('Analyzing application...'):
            try:
                # 1. Prediction
                prediction = model.predict(input_data)[0]
                prob = model.predict_proba(input_data)[0][1]

                # 2. Display Results
                if prediction == 1:
                    st.success(f'✅ Loan Approved (Probability: {prob:.2%})')
                else:
                    st.error(f'❌ Loan Rejected (Probability: {prob:.2%})')

                # 3. SHAP Explanation
                st.subheader("Why was this decision made?")

                if hasattr(model, 'named_steps'):
                    classifier = model.named_steps['classifier']
                    preprocessor = model.named_steps['preprocessor']
                
                    input_transformed = preprocessor.transform(input_data)
                
                    try:
                        feature_names = preprocessor.get_feature_names_out()
                    except AttributeError:
                        feature_names = [f"Feature {i}" for i in range(input_transformed.shape[1])]

                    # Calculate SHAP values
                    explainer = shap.TreeExplainer(classifier)
                    shap_values = explainer.shap_values(input_transformed)
            
                    if isinstance(shap_values, list):
                         shap_val_to_plot = shap_values[1] 
                    else:
                         shap_val_to_plot = shap_values

                    st_shap(shap.force_plot(
                        explainer.expected_value, 
                        shap_val_to_plot[0,:], 
                        input_transformed[0,:],
                        feature_names=feature_names
                    ))
                    
                    st.caption("Red features push the score higher (Approved), Blue features push it lower (Rejected).")
                    
                else:
                    st.warning("Model structure is not a standard Pipeline; cannot generate SHAP explanations.")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                # Useful for debugging:
                # st.write(input_data)
    else:
        st.error("Model not loaded.")

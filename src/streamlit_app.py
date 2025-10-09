import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

# Load the model from HF repo 
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Ravichandrachilde/loan-prediction-XGB",
        filename="loan_model.joblib"
    )
    return joblib.load(model_path)

model = load_model()

# UI
st.title('Loan Approval Prediction')

st.info("""
**Quick Notes:**
- **Credit History**: Select 1 for good credit (e.g., repaid loans on time), 0 for poor/no history.
- **Loan Amount Term**: Enter in **months** (e.g., 360 for 30 years).
""")

gender = st.selectbox('Gender', ['Male', 'Female'])
married = st.selectbox('Married', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
applicant_income = st.number_input('Applicant Income', min_value=0, value=5000)
coapplicant_income = st.number_input('Coapplicant Income', min_value=0, value=0)
loan_amount = st.number_input('Loan Amount', min_value=0, value=100)
loan_amount_term = st.number_input('Loan Amount Term (months)', min_value=0, value=360)
credit_history = st.selectbox('Credit History', [1, 0])  # 1 first for intuitive default
property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])

# input into DataFrame
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

# Predict
if st.button('Predict'):
    with st.spinner('Making prediction...'):
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
    st.success(f'Loan Approval: {"Yes" if prediction == 1 else "No"}')
    st.metric("Approval Probability", f"{prob:.2%}")
    
    # SHAP explanation
    st.subheader("Why? (SHAP Explanation)")
    explainer = shap.TreeExplainer(model.named_steps['classifier'])
    input_transformed = model.named_steps['preprocessor'].transform(input_data)
    shap_values = explainer.shap_values(input_transformed)
    
    # Force plot
    plt.clf()
    shap.force_plot(explainer.expected_value, shap_values[0,:], input_transformed[0,:], 
                    feature_names=model.named_steps['preprocessor'].get_feature_names_out(), 
                    matplotlib=True, show=False)
    st.pyplot(plt.gcf())
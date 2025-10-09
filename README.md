# 🏠 Loan Approval Predictor: End-to-End ML Project with XGBoost & SHAP

**A Production-Style Data Science Project: From Data to Deployed, Interpretable Model.**

---

## 🚀 Project Summary

This is an end-to-end machine learning project that predicts home loan approval status. I independently managed the entire lifecycle, from data cleaning and feature engineering to deploying this interactive, explainable model on HuggingFace Spaces. The project uses an industry-standard stack including XGBoost, SHAP, and Docker.

- **Model:** XGBoost Classifier in a scikit-learn pipeline  
- **Explainability:** SHAP visualizations for every prediction  
- **Deployment:** Streamlit app (live on HuggingFace Spaces), model/data on HuggingFace Hub  
- **Reproducibility:** All code, data, and environment details provided

![Loan Predictor App UI](assets/UI.png)

---

## 🎯 The Business Problem
Financial institutions need to process loan applications efficiently and accurately. A manual process can be slow and prone to bias. This project aims to automate the loan approval decision-making process using a data-driven model that is not only accurate but also transparent, helping loan officers understand the "why" behind each prediction to ensure fairness and build trust.

---

## 🌟 Project Highlights

- **Shows Real Project Ownership:** I handled everything—data cleaning, EDA, feature engineering, model building, deployment, and explainability.
- **Modern ML Stack:** Uses XGBoost, scikit-learn, SHAP, Streamlit, and HuggingFace Hub—tools common in industry.
- **Interpretable Results:** Every prediction explained with SHAP, making the model a "glass box" for business users.
- **Ready to Demo:** App is live; model and dataset are public and versioned.
- **Self-Driven:** Built to bridge the gap from coursework to real-world impact, showcasing my initiative and adaptability.

---

## 📈 Results & Impact

- Achieved ~75.6% test accuracy, significantly outperforming a baseline Logistic Regression model (e.g., ~68% accuracy). This demonstrates the value of using a more complex, non-linear model for this specific business problem.

- The model provides transparent, per-prediction explanations via SHAP. This can reduce manual review time for loan officers, increase decision-making confidence, and ensure algorithmic fairness by allowing for easy inspection of key influencing factors.


---

## 📂 Project Structure

```
├── assets/
│   └── UI.png             
├── src/
│   └── streamlit_app.py
├── notebooks/
│   └── Data and Model training.ipynb
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 📊 Dataset

- Source: [Loan Prediction Problem Dataset on Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- Features: Demographics, income, credit history, and loan details
- Preprocessing: Imputed missing values (mode or median), one-hot encoded categoricals, scaled numerics

---

## 🛠️ Key Steps

1. **Exploratory Data Analysis (EDA):**  
   Visualized applicant income, loan status, and feature correlations to understand drivers of approval.

2. **Data Cleaning & Feature Engineering:**  
   Imputed missing values, one-hot encoded categoricals, and scaled numeric features for robust modeling.

3. **Modeling:**  
   - Trained XGBoost classifier in a scikit-learn pipeline
   - Evaluated using accuracy, precision, recall, and F1-score on a held-out test set

4. **Explainability:**  
   - Used SHAP to interpret and visualize model decisions for every user input
   - Added force plots and summary plots for transparent predictions

5. **Deployment:**  
   - Streamlit app for user-friendly predictions and explanations
   - Model and dataset hosted on HuggingFace for open access and reproducibility

---

## 🎯 Try It Yourself

- **Live Demo:** [HuggingFace Spaces App](https://huggingface.co/spaces/Ravichandrachilde/loan-predictor-XAI)
- **Model Weights:** [HuggingFace Model Repo](https://huggingface.co/Ravichandrachilde/loan-prediction-XGB)
- **Dataset:** [HuggingFace Dataset Repo](https://huggingface.co/datasets/Ravichandrachilde/loan-prediction-dataset)

---

## 💻 Run Locally

1. **Clone the Repo**

    ```bash
    git clone https://github.com/Ravichandrachilde/Loan-Approval-Predictor.git
    cd loan-predictor-XAI
    ```

2. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit App**

    ```bash
    streamlit run src/streamlit_app.py
    ```

4. **(Optional) Use Docker**

    ```bash
    docker build -t loan-predictor-xai .
    docker run -p 8501:8501 loan-predictor-xai
    ```

---

##  Skills Demonstrated

- Full ML Lifecycle Management: Executed the entire machine learning workflow, from exploratory data analysis and feature engineering to model training, evaluation, and deployment.
- Proficiency in Modern MLOps & Data Science Tools: Utilized an industry-standard stack including XGBoost for modeling, SHAP for explainability, Scikit-learn for pipelining, Streamlit for UI development, and HuggingFace Hub for model/data versioning and deployment.
- Production-Ready Practices: Implemented best practices for reproducibility and collaboration, including dependency management (requirements.txt) and containerization (Dockerfile).
- Business-Focused Modeling: Translated a business need (loan approval) into a technical solution, focusing not just on accuracy but on interpretability and deployability to provide real-world value.



---

## 📬 Contact

- [HuggingFace Profile](https://huggingface.co/Ravichandrachilde)
- [LinkedIn](https://www.linkedin.com/in/ravichandrachilde/) 




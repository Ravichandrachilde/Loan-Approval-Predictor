# 🏠 Loan Approval Predictor: End-to-End ML Project with XGBoost & SHAP

**An interview-ready, production-style data science project by a 2024 graduate.**

---

## 🚀 Project Summary

Welcome! This is a complete machine learning project that predicts whether a home loan will be approved, using real-world data and best practices for transparency and reproducibility.  
As a 2024 graduate without internship experience, I built this to showcase the full DS/ML workflow and demonstrate job-ready skills—especially for entry-level data science and machine learning roles.

- **Model:** XGBoost Classifier in a scikit-learn pipeline  
- **Explainability:** SHAP visualizations for every prediction  
- **Deployment:** Streamlit app (live on HuggingFace Spaces), model/data on HuggingFace Hub  
- **Reproducibility:** All code, data, and environment details provided

---

## 🌟 Why Should You Care (as a Recruiter)?

- **Shows Real Project Ownership:** I handled everything—data cleaning, EDA, feature engineering, model building, deployment, and explainability.
- **Modern ML Stack:** Uses XGBoost, scikit-learn, SHAP, Streamlit, and HuggingFace Hub—tools common in industry.
- **Interpretable Results:** Every prediction explained with SHAP, making the model a "glass box" for business users.
- **Ready to Demo:** App is live; model and dataset are public and versioned.
- **Self-Driven:** Built to bridge the gap from coursework to real-world impact, showcasing my initiative and adaptability.

---

## 📈 Results & Impact

- Achieved **~75.6% test accuracy** predicting loan approvals.
- The model is especially strong at catching actual approvals (high recall & F1 for approvals), a typical challenge in imbalanced real-world datasets.
- Provides per-prediction explanations, making it easier for loan officers or business users to trust and act on model outputs.
- All code is clean, modular, and reproducible—perfect for technical interviews or code reviews.

---

## 📂 Project Structure

```
├── src/
│   └── streamlit_app.py     # Streamlit UI with SHAP explanations
├── notebooks/
│   └── train_and_explain.ipynb  # EDA, training, SHAP analysis
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
    git clone https://github.com/<your-username>/loan-predictor-XAI.git
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

## 🦾 What I Learned / Why I'm Ready

- Gained hands-on experience in the **entire ML lifecycle**: from messy data to deployed, explainable models.
- Learned modern tools (XGBoost, SHAP, Streamlit, HuggingFace Hub) used in real companies.
- Practiced code clarity, modularity, and reproducibility for teamwork and production-readiness.
- Drove the project independently, simulating a real-world business scenario—skills directly transferrable to entry-level DS/ML roles.

---

## 📬 Contact

- [HuggingFace Profile](https://huggingface.co/Ravichandrachilde)
- [LinkedIn](https://www.linkedin.com/in/ravichandrachilde/) *(edit if needed)*

---

> **Built with determination and curiosity, by a 2024 DS/ML graduate seeking opportunities.**

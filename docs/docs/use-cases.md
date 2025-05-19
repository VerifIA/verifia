# Use Case Gallery

Welcome to the *VerifIA* **Use Case Gallery**—four end‑to‑end examples demonstrating how *VerifIA* can be applied to both regression and classification problems. Each use case includes a Jupyter notebook that guides you through:

- Wrapping a trained model  
- Loading a predefined domain YAML or generating one via AI  
- Running rule‑consistency verification  
- Interpreting and exporting results  

By exploring these, you’ll see how *VerifIA* adapts across data types, frameworks, and verification goals.

---

## Regression Use Cases

### 1. Compressive Strength Regression (TensorFlow)

Verify a TensorFlow regression model trained on concrete compressive strength data. This guide demonstrates:

- Hyperparameter tuning with **scikit-optimize**  
- Manual & AI‑powered domain generation  
- Rule‑based verification with *VerifIA*  

**Artifacts:**

- [Dataset (CSV)](data/compressive_strength.csv)  
- [Application-specific docs (PDF)](documents/compressive_strength/)  
- [Domain YAML](domains/compressive_strength.yaml)  
- [Notebook](notebooks/compressive_strength.ipynb)  

---

### 2. House Pricing Prediction (CatBoost)

Verify a CatBoost regression model forecasting house prices. You will learn:

- Bayesian tuning with **BayesSearchCV**  
- Wrapping CatBoost via `CBModel`  
- Loading or generating a domain config  
- Verifying rule‑consistency with *VerifIA*  

**Artifacts:**

- [Dataset (CSV)](data/house_pricing.csv)  
- [Application-specific docs (PDF)](documents/house_pricing/)  
- [Domain YAML](domains/house_pricing.yaml)  
- [Notebook](notebooks/house_pricing.ipynb)  

---

## Classification Use Cases

### 3. Hotel Cancellation Prediction (Scikit‑Learn)

Verify a Scikit‑Learn pipeline (SVC & Random Forest) for predicting hotel cancellations. This walkthrough covers:

- Building preprocessing pipelines (`StandardScaler`, `OneHotEncoder`)  
- Tuning with `RandomizedSearchCV` & `BayesSearchCV`  
- Wrapping via `SKLearnModel`  
- Domain config & rule‑consistency verification  

**Artifacts:**

- [Dataset (CSV)](data/hotel_cancel.csv)  
- [Application-specific docs (PDF)](documents/hotel_cancel/)  
- [Domain YAML](domains/hotel_cancel.yaml)  
- [Notebook](notebooks/hotel_cancel.ipynb)  

---

### 4. Loan Eligibility Classification (XGBoost)

Verify an XGBoost classifier for loan repayment prediction. In this example you will:

- Perform Bayesian hyperparameter tuning (`BayesSearchCV`)  
- Wrap using `XGBModel`  
- Generate or load domain configurations  
- Run rule‑consistency checks  

**Artifacts:**

- [Dataset (CSV)](data/loan_eligibility.csv)  
- [Application-specific docs (PDF)](documents/loan_eligibility/)  
- [Domain YAML](domains/loan_eligibility.yaml)  
- [Notebook](notebooks/loan_eligibility.ipynb)  

---

## Why Use Cases?

- **Hands‑on guidance:** Detailed notebooks covering each step.  
- **Framework diversity:** TensorFlow, CatBoost, Scikit‑Learn, XGBoost.  
- **Balanced scenarios:** Two regression and two classification examples.  
- **Best practices:** Tips on tuning, domain setup, and result interpretation.

---

> **Next Steps:**  
> 1. Choose the use case matching your problem type.  
> 2. Clone the corresponding notebook and run it locally.  
> 3. Adapt the patterns to verify your own models with *VerifIA*.

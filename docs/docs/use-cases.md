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

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Nsxkw2t29R7E9mBRT2tZmS_cP1cWOcno?usp=sharing)

**Artifacts:**

- [Dataset (CSV)](https://www.verifia.ca/assets/use-cases/data/concrete_compressive_strength.csv)  
- [Domain YAML](https://www.verifia.ca/assets/use-cases/domains/concrete_compressive_strength.yaml)  
- [Application-specific docs (Zipped PDFs)](https://www.verifia.ca/assets/use-cases/documents/concrete_compressive_strength.zip)   

---

### 2. House Pricing Prediction (CatBoost)

Verify a CatBoost regression model forecasting house prices. You will learn:

- Bayesian tuning with **BayesSearchCV**  
- Wrapping CatBoost via `CBModel`  
- Loading or generating a domain config  
- Verifying rule‑consistency with *VerifIA*  

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11wzH3sDB7NesU5d_6GKmL2xNqqCA8AzA?usp=sharing)

**Artifacts:**

- [Dataset (CSV)](https://www.verifia.ca/assets/use-cases/data/house_price.csv)  
- [Domain YAML](https://www.verifia.ca/assets/use-cases/domains/house_price.yaml)  
- [Application-specific docs (Zipped PDFs)](https://www.verifia.ca/assets/use-cases/documents/house_price.zip)  

---

## Classification Use Cases

### 3. Hotel Cancellation Prediction (Scikit‑Learn)

Verify a Scikit‑Learn pipeline (SVC & Random Forest) for predicting hotel cancellations. This walkthrough covers:

- Building preprocessing pipelines (`StandardScaler`, `OneHotEncoder`)  
- Tuning with `RandomizedSearchCV` & `BayesSearchCV`  
- Wrapping via `SKLearnModel`  
- Domain config & rule‑consistency verification  

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zp1wOd7BiYW65GI0FAGRjMj4LwruXXYB?usp=sharing)

**Artifacts:**

- [Dataset (CSV)](https://www.verifia.ca/assets/use-cases/data/hotel_cancellation.csv)  
- [Domain YAML](https://www.verifia.ca/assets/use-cases/domains/hotel_cancellation.yaml) 
- [Application-specific docs (Zipped PDFs)](https://www.verifia.ca/assets/use-cases/documents/hotel_cancellation.zip)  
 
---

### 4. Loan Eligibility Classification (XGBoost)

Verify an XGBoost classifier for loan repayment prediction. In this example you will:

- Perform Bayesian hyperparameter tuning (`BayesSearchCV`)  
- Wrap using `XGBModel`  
- Generate or load domain configurations  
- Run rule‑consistency checks  

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1R8y2YSBLkUWcA25jt8cORY9cy5oLd7_m?usp=sharing)

**Artifacts:**

- [Dataset (CSV)](https://www.verifia.ca/assets/use-cases/data/loan_eligibility.csv)  
- [Domain YAML](https://www.verifia.ca/assets/use-cases/domains/loan_eligibility.yaml)  
- [Application-specific docs (Zipped PDFs)](https://www.verifia.ca/assets/use-cases/documents/loan_eligibility.zip)  
 
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

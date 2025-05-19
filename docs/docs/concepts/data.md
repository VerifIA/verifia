---
icon: octicons/database-24
---

# Data

*VerifIA*’s verification pipeline is entirely **data‑driven**. The dataset you supply becomes the **original seed** from which *VerifIA* derives novel, out‑of‑sample inputs. Evaluating your model on these derived samples lets *VerifIA* measure its generalizability beyond the labeled data by verifying consistency with your application’s domain knowledge.

---

## 1. Seed Data

- The **original seed** is **never** evaluated “as‑is.”  
- It powers the **generation** of all verification inputs.  
- You may include:
    - **Training data** (examples already seen by the model)  
    - **Validation** or **test data** (often yield more **challenging** derived inputs, since those regions tend to be less covered by the model)

*VerifIA* offers three ways to attach your seed data:

=== "A. From Local Files"

    ```python
    from verifia.verification import RuleConsistencyVerifier

    verifier = RuleConsistencyVerifier("domain_rules.yaml")
    # Supported formats: CSV, Excel, JSON, Parquet, Feather, Pickle
    verifier = verifier.on(data_fpath="path/to/data.csv")
    ```

    | Format          | Extension         |
    | --------------- | ----------------- |
    | CSV          | `.csv`     |
    | Excel | `.xls`, `.xlsx`   |
    | JSON         | `.json`     |
    | Parquet      | `.parquet` |
    | Feather      | `.feather` |
    | Pickle          | `.pkl`  |

=== "B. From pandas DataFrame"

    If your data comes from SQL, NoSQL, or APIs, load it into a DataFrame and pass it directly:

    ```python
    from verifia.verification import RuleConsistencyVerifier
    from your_code import load_my_data

    verifier = RuleConsistencyVerifier("domain_rules.yaml")
    df = load_my_data()  # e.g. pd.read_sql, API call, etc.
    verifier = verifier.on(dataframe=df)
    ```

=== "C. Using VerifIA’s Dataset Object"

    To handle metadata (e.g., categorical features), wrap your DataFrame in `Dataset`:

    ```python
    from verifia.context.data import Dataset
    from verifia.verification import RuleConsistencyVerifier

    # 1. Load raw data
    import pandas as pd
    df = pd.read_csv("path/to/data.csv")

    # 2. Define metadata
    target = "predicted_label"
    features = ["feat_1", "feat_2", ...]
    cat_features = ["cat_feat_1", "cat_feat_2"]

    # 3. Create Dataset
    ds = Dataset(df, target, features, cat_features)

    # 4. Initialize verifier and attach Dataset
    verifier = RuleConsistencyVerifier("domain_rules.yaml")
    verifier = verifier.on(dataset=ds)
    ```

---

## 2. Derived Inputs

*VerifIA* applies your configured search strategy (population size, iterations) and your defined domain (variables' ranges,constraints, and rules) **to the original seed**—to explore new in‑domain datapoints. This approach probes beyond perfectly in‑distribution regions represented by labeled examples.

---

## 3. Out‑of‑Domain Removal

1. **Validate** each seed row against its feature metadata (ranges, types, constraints).  
2. **Remove** rows with aberrant values (out‑of‑range, undefined) to prevent generation errors.  
3. **Report** includes the count of removed out-of-domain seeds. 

!!! warning

    A high **out‑of‑domain** count can indicate:

    - Noisy or invalid seed data  
    - Incomplete or overly restrictive metadata  

---

## 4. Misprediction Exclusion

To avoid biasing verification by known model errors, *VerifIA* excludes any seed row where the model already fails:

- **Misclassification:** predicted class ≠ ground truth  
- **High Regression Error:** |prediction – ground truth| > allowed tolerance 

!!! failure "High Misprediction Rate"

    - **Issue:** A high number of exclusions signals that the model needs further optimization before advanced behavioral tests.  
    - **Action:** Retrain or fine‑tune until mispredictions on validation/test examples drop to an acceptable level, then proceed with *VerifIA*'s verifications.

The report will show the **count of excluded, mispredicted seeds**.

---

## 5. Seed Diversity

- A **diverse, well‑dispersed** seed lets *VerifIA* cover more of the input space.  
!!! note "Avoid sampling original seeds in narrow input space regions"

    - Visualize feature histograms, pairwise plots, or scatter plots to ensure broad coverage.  
    - Perform unsupervised clustering (e.g., k‑means, DBSCAN) or compute pairwise distance metrics.  

---

## 6. Verification Budget & Runtime

- Each seed row spawns many derived trials (population size × iterations × rules).  
- **Adding seeds ≠ linear runtime**—more seeds can dramatically increase run time.  
- **Balance** seed diversity against your compute budget; tune search parameters accordingly.

---

!!! example "Supported Data Types"
    - **Current:** Tabular data  
    - **Future:** Images, natural‑language text, audio, and beyond  

---

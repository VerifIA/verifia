# Tutorial 1: Rule‑Based Dummy Regression Verification

Learn how to use *VerifIA* to verify a synthetic regression model against rule-based expectations. This tutorial covers:

* Generating synthetic data with known feature-target relationships
* Training a scikit-learn regression pipeline
* Defining domain rules in YAML
* Wrapping the model and running the `RuleConsistencyVerifier`
* Inspecting and exporting verification results

---

## 1. Setup & Imports

Import standard data-science libraries and VerifIA modules.

```python
import logging
import pandas as pd
import tempfile

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

from verifia.models import SKLearnModel, build_from_model_card
from verifia.verification.verifiers import RuleConsistencyVerifier
from verifia.verification.results import RulesViolationResult
```

!!! tip "Logging"

    You can adjust the logging level to DEBUG to see detailed verifier steps:

    ```python
    logging.basicConfig(level=logging.INFO)
    ```

---

## 2. Generate Synthetic Data

Create a 3-feature regression dataset where:

* `pos_imp_1` and `pos_imp_2` are positively correlated with the target
* `zero_imp` is uninformative

```python
RAND_SEED = 0
feature_names = ["pos_imp_1", "zero_imp", "pos_imp_2"]

def make_data():
    X, y, _ = make_regression(
        n_samples=1000,
        n_features=3,
        n_informative=2,
        noise=3,
        bias=0.1,
        coef=True,
        random_state=RAND_SEED
    )
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return train_test_split(df, train_size=0.8, random_state=RAND_SEED)

train_df, test_df = make_data()
```

??? note

    * We fix the random seed for reproducibility.
    * Split data into 80% train / 20% test.

---

## 3. Train the Model

Construct and fit a scikit-learn pipeline:

```python
pipeline = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=2),
    LinearRegression()
)

pipeline.fit(train_df[feature_names], train_df["target"])
pred = pipeline.predict(test_df[feature_names])
print("RMSE:", mean_squared_error(test_df["target"], pred, squared=False))
print("MAPE:", mean_absolute_percentage_error(test_df["target"], pred))
```

??? note

    * We use degree-2 polynomial features to capture nonlinear relationships.
    * Metrics on the test set give baseline performance.


---

## 4. Define Domain Rules

We encode our known correlations into the domain config YAML:

```yaml
variables:
  pos_imp_1:
    type: FLOAT
    range: [-3, 3]
  pos_imp_2:
    type: FLOAT
    range: [-3, 3]
  zero_imp:
    type: FLOAT
    range: [-3, 3]
  target:
    type: FLOAT
    range: [-100, 100]
    insignificant_variation: 0.05

constraints: {}

rules:
  R_inc_pure:
    premises: { pos_imp_1: inc, pos_imp_2: inc }
    conclusion: { target: nodec }
  R_dec_pure:
    premises: { pos_imp_1: dec, pos_imp_2: dec }
    conclusion: { target: noinc }
  R_inc_all:
    premises: { pos_imp_1: inc, pos_imp_2: inc, zero_imp: var }
    conclusion: { target: nodec }
  R_dec_all:
    premises: { pos_imp_1: dec, pos_imp_2: dec, zero_imp: var }
    conclusion: { target: noinc }
```

---

## 5. Wrap & Verify

1. **Temporary YAML file** (for demo):

  Save the YAML to `domain_rules.yaml` or load it via a temporary file as shown below.

   ```python
   tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml")
   tmp.write(yaml_string)
   tmp.close()
   domain_fpath = tmp.name
   ```

2. **Wrap model** with metadata:

   ```python
   model_wrapper: SKLearnModel = build_from_model_card({
       "name": "synthetic_reg",
       "version": "1",
       "type": "regression",
       "description": "Synthetic data regressor",
       "framework": "sklearn",
       "feature_names": feature_names,
       "target_name": "target",
       "local_dirpath": "../models"
   }).wrap_model(pipeline)
   ```

3. **Instantiate verifier & link data**:

   ```python
   verifier = RuleConsistencyVerifier(domain_fpath)
   verifier.verify(model_wrapper).on(test_df)
   ```

4. **Run with PSO** (example):

   ```python
   result: RulesViolationResult = (
       verifier.using("PSO")
               .run(pop_size=100, max_iters=10, orig_seed_ratio=1.0)
   )
   ```

!!! success

    ✅ Verification complete!  
      - View Report via `result.display()`  

---

## 6. Export Report

```python
result.save_as_html("synthetic_reg_report.html")
```

!!! tip

    Analyze the HTML report to grasp the different provided metrics.

---

*Next:* [Tutorial #2: Manual vs. AI‑Powered Domain Generation](tuto_2.md)

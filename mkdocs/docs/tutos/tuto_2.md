# Tutorial 2: Manual vs. AI‑Powered Domain Generation

Explore two methods for generating a domain configuration on the California housing dataset:

1. **Manual Domain Dictionary** based on feature intuition.
2. **AI‑Powered Generation** using VerifIA’s `DomainGenFlow`.

---

## 1. Setup & Imports

Import core libraries and VerifIA modules:

```python
import logging
from dotenv import load_dotenv
load_dotenv()

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

from verifia.models import SKLearnModel, build_from_model_card
from verifia.verification.verifiers import RuleConsistencyVerifier
from verifia.verification.results import RulesViolationResult
from verifia.generation import DomainGenFlow
```

!!! tip "Environment"

    Store secrets (e.g., OpenAI key, GPT settings) in a `.env` file and call `load_dotenv()` to configure.

---

## 2. Constants & Data Loading

```python
RAND_SEED = 0
MODELS_DIRPATH = "../models"

housing = fetch_california_housing(as_frame=True)
housing_df = housing.frame
feature_names = housing.feature_names
target_name = housing.target_names[0]

train_df, test_df = train_test_split(
    housing_df, train_size=0.8, random_state=RAND_SEED
)
```

??? note

    * 80/20 train-test split for consistent evaluation.
    * Use `feature_names` and `target_name` directly from dataset metadata.

---

## 3. Train Multiple Models

=== "Polynomial Regression"

    ```python
    poly_reg = make_pipeline(
        StandardScaler(), PolynomialFeatures(degree=2), LinearRegression()
    )
    poly_reg.fit(train_df[feature_names], train_df[target_name])
    pred = poly_reg.predict(test_df[feature_names])
    print("RMSE:", mean_squared_error(test_df[target_name], pred, squared=False))
    print("MAPE:", mean_absolute_percentage_error(test_df[target_name], pred))
    ```

=== "Decision Tree"

    ```python
    tree_reg = make_pipeline(
        StandardScaler(), DecisionTreeRegressor(random_state=RAND_SEED)
    )
    tree_reg.fit(train_df[feature_names], train_df[target_name])
    pred = tree_reg.predict(test_df[feature_names])
    print("RMSE:", mean_squared_error(test_df[target_name], pred, squared=False))
    print("MAPE:", mean_absolute_percentage_error(test_df[target_name], pred))
    ```

=== "Random Forest"

    ```python
    forest_reg = make_pipeline(
        StandardScaler(), RandomForestRegressor(random_state=RAND_SEED)
    )
    forest_reg.fit(train_df[feature_names], train_df[target_name])
    pred = forest_reg.predict(test_df[feature_names])
    print("RMSE:", mean_squared_error(test_df[target_name], pred, squared=False))
    print("MAPE:", mean_absolute_percentage_error(test_df[target_name], pred))
    ```

=== "MLP Regressor"

    ```python
    mlp_reg = make_pipeline(
        StandardScaler(), MLPRegressor(hidden_layer_sizes=(128,64,32), random_state=RAND_SEED)
    )
    mlp_reg.fit(train_df[feature_names], train_df[target_name])
    pred = mlp_reg.predict(test_df[feature_names])
    print("RMSE:", mean_squared_error(test_df[target_name], pred, squared=False))
    print("MAPE:", mean_absolute_percentage_error(test_df[target_name], pred))
    ```

---

## 4. Wrap Model with VerifIA

```python
model_card = {
    "name": "CHPrice_skl_poly_regressor",
    "version": "1",
    "type": "regression",
    "description": "Predict California housing prices",
    "framework": "sklearn",
    "feature_names": feature_names,
    "target_name": target_name,
    "local_dirpath": MODELS_DIRPATH
}

model_wrapper: SKLearnModel = build_from_model_card(model_card)
    .wrap_model(poly_reg)  # swap for tree_reg, forest_reg, mlp_reg
```

---

## 5. Create Domain Configuration

### Option A – Manual Domain Dictionary

You can manually create a simple domain dictionary. In this example, a dictionary is built where each variable is defined based on the features from the California housing dataframe. A sample constraint (e.g., a ratio between average bedrooms and rooms) and a rule (R1) are included. You can further customize and extend this dictionary with additional rules as needed.

```python
domain_cfg_dict = {
    "variables":{
        col: {
            "type": "INT" if (is_int := (housing_df[col] == housing_df[col].round()).all()) else "FLOAT",
            "range": (housing_df[col].astype(int) if is_int else housing_df[col]).agg(['min', 'max']).tolist()
        }
        for col in housing_df.columns
    },
    "constraints":{
        "C1": {
                "description":"", 
                "formula": "AveBedrms/AveRooms > 0.5"
            }
    },
    "rules":{
        "R1": {
               "description": "",
               "premises": {"AveRooms":"inc", "AveBedrms":"inc", "HouseAge": "dec"},
               "conclusion": {"MedHouseVal":"inc"}
            }
    }
}
domain_cfg_dict["variables"]['MedHouseVal']['insignificant_variation'] = 0.15 # expect 15% of error as acceptable
```

!!! tip

    Build `domain_cfg_dict` by inspecting `housing_df.describe()` for sensible ranges.


### Option B – AI‑Powered Domain Generation

Alternatively, you can leverage VerifIA’s `DomainGenFlow` to generate a domain dictionary automatically. By providing the dataframe and a description (here, the dataset’s description from `housing.DESCR`), the tool generates a domain configuration using AI. 

```python
genflow = DomainGenFlow()
genflow.load_ctx(
    dataframe=housing_df,
    db_str_content=str(housing.DESCR),
    model_card=model_card
)
domain_cfg_dict = genflow.run()
```

!!! note

    Customize GPT via `VERIFIA_GPT_NAME` and `VERIFIA_GPT_TEMPERATURE` environment variables.

---

## 6. Run Rule Consistency Verification

```python
verifier = RuleConsistencyVerifier(domain_cfg_dict)
verifier.verify(model_wrapper).on(test_df)
result: RulesViolationResult = (
    verifier.using("GA")
            .run(pop_size=50, max_iters=10, orig_seed_size=100)
)
```

* **Algorithm:** choose from RS, GA, PSO, etc.
* **Parameters:** `pop_size` (# candidates), `max_iters` (search steps), `orig_seed_size` (seed ratio).

!!! success

    ✅ Verification complete!  
    - View Report via `result.display()`  

---

## 7. Export Results & Artifacts

```python
result.save_as_html("CHPrice_skl_poly_report.html")
model_wrapper.save_model()
model_wrapper.save_model_card("CHPrice_model_card.yaml")
```

!!! tip

    Use meaningful filenames (e.g., `tree`, `forest`, `mlp`) to identify model variants easily.

---

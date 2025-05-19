---
icon: material/test-tube
---

# Model

*VerifIA* performs a **domain‑aware verification** on a model that has already passed all standard evaluation steps, as described in the following pre‑Verification requirements. 

Your model must be:

- **Optimized**  
    - Model parameters are carefully trained and fine-tuned.

- **Validated**  
    - All modeling decisions are evaluated on a **validation set** (e.g., hyperparameters).

- **Tested**  
    - Performance is assessed on **unseen test sets**, used as a proxy for future data.

!!! success

    Only once these evaluation steps yield satisfactory results should you run *VerifIA*’s domain‑aware verification.

## 1. Staging Model Setup

### 1.1 Build the Wrapper

Select the appropriate wrapper or use a model card helper:

=== "A. Direct Instantiation"

    ```python
    from verifia.models import SKLearnModel, build_from_model_card

    # Example metadata
    feature_names = ["feat_1", "feat_2"]
    target_name = "predicted_label"
    cat_features = ["cat_feat_1"]

    model_wrapper = SKLearnModel(
        name="my_model",
        version="1",
        model_type="regression",
        feature_names=feature_names,
        target_name=target_name,
        local_dirpath="./models",
        cat_feature_names=cat_features
    )
    ```

=== "B. From Model Card"

    ```python
    from verifia.models import SKLearnModel, build_from_model_card

    # Example metadata
    feature_names = ["feat_1", "feat_2"]
    target_name = "predicted_label"
    cat_features = ["cat_feat_1"]

    model_card = {
    "name": "skl_model",
    "version": "1",
    "type": "regression",
    "framework": "sklearn",
    "feature_names": feature_names,
    "target_name": target_name,
    "cat_feature_names": cat_features,
    "local_dirpath": "./models"
    }
    model_wrapper = build_from_model_card(model_card)
    ```

Supported frameworks: sklearn, lightgbm, catboost, xgboost, pytorch, tensorflow.

### 1.2 Load the Model

=== "A. Wrap an in-memory object"

    ```python
    model_obj = ...  # optimized model object
    model_wrapper.wrap_model(model_obj)
    ```

=== "B. Load from local file"

    ```python
    # If file path is None, VerifIA uses default: {name}-{version}.{ext}
    model_wrapper.load_model(model_fpath=None)
    ```

    Default extensions: pkl, txt, cb, json, pth, keras.

=== "C. Load from registry"

    ```python
    # After setting platform env vars
    model_wrapper.load_model_from_registry()
    ```

    Supported platforms: MLflow, Comet ML, WandB.

---

### 1.3. Attach Model to Verifier

```python
from verifia.verification import RuleConsistencyVerifier
verifier = RuleConsistencyVerifier("domain_rules.yaml")
verifier = verifier.verify(model=model_wrapper)
```

??? info "Auto-load"

    You may also skip manual loading:

    ```python
    # Pass model card directly
    verifier = verifier.verify(model_card=model_card)
    ```
    
    *VerifIA* auto-loads from registry or local if needed.

---

## 2. Statistical Evaluation Caveats

Standard metrics (accuracy, RMSE, F1, etc.) depend heavily on your data’s:

- **Correctness**  
    - Low noise, high precision  

- **Coverage**  
    - Diverse, comprehensive sampling of the input distribution  

!!! warning "Trade-off"

    Achieving both can be challenging in real‑world settings.

---

## 3. Why Domain‑Aware Verification?

Traditional statistical evaluation assumes:

  1. **IID inputs**  
  2. **Representative test data** covering all foreseeable cases  

*VerifIA* goes beyond these assumptions by:

  - Generating **novel, out‑of‑sample inputs**  
  - Verifying model behavior against **application‑specific domain knowledge**  
  - Maximizing coverage of the model’s **input space**

!!! success "Expanded coverage"

    *VerifIA* helps explore regions of the input space that training and test sets may miss.  

!!! success "Behavior consistency"

    Domain rules expose inconsistencies, alerting you to harmful noise or bias in training data. 

!!! warning "Original Seed Quality"

    *VerifIA* still relies on your **original seed data** to derive new test cases. Poor data quality or insufficient coverage can lead to misleading or unreliable verification results.  
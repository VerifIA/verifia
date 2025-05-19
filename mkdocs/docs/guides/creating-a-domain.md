# Creating a Domain YAML File

This guide shows you how to elaborate the **domain specification** that *VerifIA* uses to generate and validate derived inputs. For a deeper conceptual overview of “domain” in *VerifIA*, see [Concepts → Domain](/concepts/domain).

---

## 1. Why a Domain File?

*VerifIA* is **domain‑aware**: it requires a priori domain knowledge related to your application. 

Encoding that knowledge in YAML lets *VerifIA*:

- **Derive novel inputs** beyond your dataset  
- **Enforce feasibility** (via constraints)  
- **Assert behavioral rules** on model outputs  

---

## 2. File Structure

Your domain file has three top‑level sections:

```yaml

variables:    # define inputs & outputs  
constraints:  # enforce feasibility on input variables only  
rules:        # assert expected behavior  
```

---

### 2.1. Section: variables

Define **raw features** and **model targets** on their real‑world scale—no preprocessing required.

| Key                       | Description                                                                                                          |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `description`             | Human‑readable label for the feature or target                                                                       |
| `type`                    | `INT` / `FLOAT` / `CAT`                                                                                              |
| `range`                   | `[min, max]` for numeric variables                                                                                   |
| `values`                  | `["v1","v2",…]` for categorical variables (order matters if ordinal)                                                 |
| `formula` *(optional)*    | Python expression to compute one variable from others                                                                |
| `variation_limits`        | `[min_ratio, max_ratio]` percent change when deriving <br> e.g. `[0.025,0.5]` = 2.5%–50% variation                      |
| `insignificant_variation` | Float `0.0–1.0` defining:<br>• **Input noise** to reveal brittleness<br>• **Output tolerance** for regression errors |

!!! note

    * Use `variation_limits` to control drift from the seed.
    * Use `insignificant_variation` to inject white‑noise on inputs or allow tolerable error on outputs.

#### Example variables block

```yaml
variables:
  age:
    description: Customer age in years
    type: INT
    range: [18, 100]
    variation_limits: [0.01, 0.10]         # 1%–10% change
    insignificant_variation: 0.01         # 1% noise

  income:
    description: Annual income in USD
    type: FLOAT
    range: [0, 1e6]
    formula: 12 * monthly_income          # derived from another variable

  churn_probability:
    description: Model’s output probability
    type: FLOAT
    range: [0.0, 1.0]
    insignificant_variation: 0.02         # 2% tolerated error
```

---

### 2.2. Section: constraints

Define **feasibility constraints** over **input variables only**—never reference the model’s output or target. These formulas capture interdependencies that make certain input combinations invalid. *VerifIA* discards any derived point violating these constraints.

| Key           | Description                                          |
| ------------- | ---------------------------------------------------- |
| `description` | Human‑readable explanation of the constraint         |
| `formula`     | Python expression involving **only input variables** |

#### Example constraints block

```yaml
constraints:
  max_income_age_ratio:
    description: Income must not exceed 100,000 × age
    formula: "income <= 100000 * age"
```

!!! note

    Constraints must **not** include the target/output variable—use rules for output behavior assertions.

---

### 2.3. Section: rules

Rules assert **relative expectations** on model outputs when inputs change. They consist of:

1. **Premises**: how inputs vary
2. **Conclusion**: expected output response

#### 2.3.1 Example rules block

```yaml
rules:
  high_income_reduces_churn:
    description: 
      If income increases (age constant), churn probability should decrease
    premises:
      income: inc
      age: cst
    conclusion:
      churn_probability: dec
```

#### 2.3.2 Premises: Specifying Input Variations 

| Directive         | Meaning                        | Details & Examples                                             |
| ----------------- | ------------------------------ | -------------------------------------------------------------- |
| `inc`             | Increase                       | Value is **strictly greater** than its original seed value.    |
| `dec`             | Decrease                       | Value is **strictly less** than its original seed value.       |
| `cst`             | Constant                       | Value **remains unchanged**.                                   |
| `var`             | Vary                           | May **change** freely within variation limits.   |
| `eq("v")`         | Equal to specific value        | Must be set **exactly** to `"v"`.        |
| `noeq("v")`       | Not equal to specific value    | Must **not** be `"v"`.                |
| `in("v1","v2")`   | One of a set of allowed values | Must be **one of** `"v1"` or `"v2"`. |
| `noin("v1","v2")` | None of a set of values        | Must **not** be `"v1"` nor `"v2"`.  | 

!!! note 
    * Combine multiple premises to constrain multi‑dimensional variations.
    * Omitted variables default to `cst`.

#### 2.3.3 Conclusion: Expected Output Response

  | Directive | Meaning                                                     | When to Use                                                 |
  | --------- | ----------------------------------------------------------- | ----------------------------------------------------------- |
  | `inc`     | Output should **increase** relative to seed prediction.     | E.g., “More feature\_X → Higher risk\_score.”               |
  | `dec`     | Output should **decrease** relative to seed prediction.     | E.g., “Higher price → Lower demand.”                        |
  | `cst`     | Output should **remain unchanged**.                         | E.g., “Changing log level does not affect accuracy.”        |
  | `noinc`   | Output should **not increase** <br> -> may decrease or stay flat. | Use when increases are disallowed but decreases acceptable. |
  | `nodec`   | Output should **not decrease** <br> -> may increase or stay flat. | Use when decreases are disallowed but increases acceptable. |

!!! note
    Rule-based verification consists of comparing model predictions on derived inputs against these expectations. Violations count toward consistency metrics.

---

## 3. Full YAML Template

Use this template as `domain.yaml`—replace placeholders with your actual domain definitions:

```yaml
variables:
  feature_1:
    description: Human‑readable description
    type: FLOAT
    range: [0.0, 100.0]
    variation_limits: [0.01, 0.20]
    insignificant_variation: 0.02

  feature_2:
    description: Categorical feature
    type: CAT
    values: ["low","medium","high"]

  derived_feature:
    description: Computed feature
    type: FLOAT
    formula: "2 * feature_1 + 5"

  target:
    description: Model’s output
    type: FLOAT
    range: [0.0, 1.0]
    insignificant_variation: 0.05

constraints:
  valid_income_age:
    description: Income cannot exceed 100 × age
    formula: "income <= 100 * age"

  non_negative_balance:
    description: Balance must be ≥ 0
    formula: "balance >= 0"

rules:
  demand_vs_price:
    description: When price increases, demand should decrease
    premises:
      price: inc
      season: cst
    conclusion:
      demand: dec

  risk_vs_age:
    description: Risk score should not decrease when age increases
    premises:
      age: inc
    conclusion:
      risk_score: noinc
```

---
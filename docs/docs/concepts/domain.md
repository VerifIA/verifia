---
icon: simple/roundcube
---

# Domain

*VerifIA* is a **domain‑aware verification** tool that assesses a predictive model’s consistency with your application’s domain knowledge. Instead of verifying only on existing labeled examples, *VerifIA* **derives novel inputs** from your seed dataset—expanding input‑space coverage and building confidence that the model will behave correctly under real‑world, future conditions.

---

## 1. Initializing the Verifier’s Domain

*VerifIA*’s `RuleConsistencyVerifier` requires your **domain configuration** up front. To create yours, Follow the detailed guidelines in [Domain Creation Guide](/guides/creating-a-domain). Once it is done, Supply it either as a YAML file path or as an in‑memory dictionary:

=== "A. From YAML File"

    ```python
    from verifia.verification import RuleConsistencyVerifier

    # Load rules from a domain YAML file
    verifier = RuleConsistencyVerifier("path/to/domain_rules.yaml")
    ```

=== "B. From Python Dictionary"

    ```python
    from verifia.verification import RuleConsistencyVerifier

    # Define your domain config in code
    domain_cfg = {
      "variables":    { … },
      "constraints":  { … },
      "rules":        { … }
    }

    # Instantiate verifier directly from dict
    verifier = RuleConsistencyVerifier(domain_cfg)
    ```

## 2. Application Domain Knowledge

Your **domain knowledge** includes all the a priori information you possess about your application—particularly expectations around how your input (and output) variables are distributed.

- **Documented rules or formulas**  
- **Simulator logic** in legacy systems  
- **Expert heuristics** and common‑sense insights  

!!! note

    You gather the domain knowledge; *VerifIA* then helps you summarize it into a compute‑friendly format and leverages it to generate and validate novel test cases.

---

## 3. Domain Variables 

*VerifIA* works on **raw** features and outputs—no preprocessing or feature engineering required. 

For each variable you must specify:

### 3.1. Definition

- **Type:** `INT`, `FLOAT`, or `CAT`  
- **Domain:**  
    - *Numeric* → `[min, max]` range  
    - *Categorical* → list of allowed values (order matters if ordinal)  
- **Computed formula** (optional): derive one variable from others  

_All definitions use real‑world scales so that derived inputs remain plausible._

To expose model brittleness, *VerifIA* perturbs inputs within two configurable thresholds:

### 3.2. Variation limits 
    
- **Range:** `[min_ratio, max_ratio]`  
    - *min_ratio* ensures changes are large enough to trigger detectable effects  
    - *max_ratio* prevents unrealistic drift from the original seed 

!!! tip

    Adjust `max_ratio` higher for robustness testing, lower for strict domain consistency.

### 3.3. Insignificant variation 

- **Ratio** within (`0.0–1.0`)  
    - *Inputs:* injects white‑noise to reveal fragile decision nonlinear boundaries
    - *Outputs:* defines acceptable prediction error when evaluating regression targets  

!!! Example

    A 1% noise on a float input can uncover hidden biases where minor input shifts cause outsized output changes.

---

## 4. Feasibility Constraints

Individual variable ranges do not capture interdependencies. **Constraints** are mathematical conditions over multiple variables. *VerifIA* uses them to define and enforce the feasible input space:

- Express relationships such as “`feature_A + feature_B <= 100`”  
- Discard any derived datapoint violating these formulas  

!!! abstract "Insight"

    Well‑defined constraints prevent generation of *impossible* or *semantically-invalid* inputs.

---

## 5. Behavioral Rules

The core of domain verification lies in **rules** that assert how the model’s **output** should respond when inputs change:

- **Premises:** describe input variations (`increase`, `decrease`, `stall`, or categorical conditions: `equal`, `in`, etc.)  
- **Conclusions:** assert output direction (`increase`, `decrease`, `stall`) or allowed stability (`no increase`, `no decrease`)  

*VerifIA* uses these rules to:

1. **Derive new inputs** consistent with domain space and constraints  
2. **Compute expected output behavior** without needing groundtruth labels  

!!! abstract "Insight"

    Unlike statistical tests, you do not need to know exact target values for the novel inputs—rules express **relative** expectations (e.g., “if price goes up, demand should go down”).
    
    - **No ground‑truth targets** required for derived inputs
    - **Seed validation:** original seeds must be in-domain and correctly handled by the model
    - **Consistency‑Only Verification:** complements standard statistical evaluation (accuracy, RMSE) by ensuring domain‑level consistency beyond IID assumptions  

---

!!! tip "Tips"

    - **Balance variation**: too little change hides problems; too much change explores unrealistic scenarios.  
    - **Iterate**: refine rules and constraints based on initial findings to target subtle behavioral gaps.

---

## 6. Guided Subspace Exploration

Applying every rule to every seed row defines a **constrained subspace** of potential inputs. 

Exhaustive enumeration is unpractical, so *VerifIA* employs **population‑based metaheuristics** to reveal **inconsistencies** in model behavior regardless of how rare they may be.

!!! danger "Be Careful"
    
    - Tune the run parameters (population size, max iterations) to balance discovery power against compute budget.
    - Poorly-defined rules or constraints can lead to false positives or missed inconsistencies.  
---


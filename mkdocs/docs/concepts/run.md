---
icon: material/checkbox-marked-circle-auto-outline
---

# Run

A **run** is a single *VerifIA* execution over your **domain** rules that searches for behavioral inconsistencies of your **model** using a selected original seeds. It is the building block of a full verification **report** that can be saved in HTML report. 

Each verification **report** is scoped to a specific **model** version and a defined **domain**, yet it aggregates all runs—across different seed subsets and search algorithms—to build cumulative evidence of the model’s consistency within that domain.

---

## 1. Run Configuration

Each run is parameterized by:

- **Population size** (`pop_size`): number of candidate inputs per generation. Check [searcher's run-wise parameters](/concepts/searcher).
- **Max iterations** (`max_iters`): evolutionary budget per seed–rule subspace. Check [searcher's run-wise parameters](/concepts/searcher).  
- **Sampling strategy** (`orig_seed_ratio` or `orig_seed_size`): randomly select a subset of seed inputs instead of using the full dataset.

!!! tip "Purpose of Sampling"

    * **Fast trials:** shorter runtimes to validate domain metadata and hyperparameters by **quick** “smoke tests”. 
    * **Variance analysis:** run multiple samples to gauge how seed diversity impacts consistency metrics
    * **Configuration tuning:** iterate on population/iteration settings without full‑data cost

---

## 2. Verification Result

*VerifIA* can **persist** the outcomes of multiple runs into a single, consolidated report—tracking both per‑run details and aggregated metrics.

---

### 2.1 Run‑Level Metrics

Each individual run records:

* **Parameters**
/

* **Searcher**

    * Algorithm name (e.g. GA, PSO)
    * Algorithm parameters

* **Run Statistics**

    * Seed rows evaluated
    * Total rules applied
    * Total derived verifications
    * Out‑of‑domain exclusions
    * Seed mispredictions

* **Performance Metrics**

    * **Consistency rate** (overall / per rule): fraction of derived inputs satisfying **all** rules
    * **Compliance rate** (overall / per rule): fraction satisfying **all** constraints
    * **Violation count** (overall / per rule): number of derived inputs breaking ≥ 1 rule
    * **Infeasible count** (overall / per rule): derived points discarded by constraints
    * **Inconsistency‑revealing seed count** (overall / per rule): seeds that generated at least one violating input
    * **Average deviation per rule**: mean magnitude by which predictions diverged from expected behavior

---

### 2.2 Report‑Level Metrics

The consolidated report aggregates across runs to provide:

* **Model Metadata**

    * Name, version, framework

* **Aggregate Statistics**

    * Total seed size
    * Total rules
    * Total verifications
    * Out‑of‑domain rate
    * Seed misprediction rate

* **Domain Compliance**

    * **Violated rules** and overall **rules consistency ratio**
    * **Unsatisfied constraints** and overall **constraints compliance ratio**
    * **Frequency of inconsistency‑revealing seeds**

!!! abstract "Insight"
    High consistency with low deviation indicates strong domain alignment. Low consistency or high deviation highlights model weaknesses or gaps in domain definitions.

---

## 5. Aggregating & Incremental Domains

* **Multiple runs** can vary in seed subsets, algorithms, or parameters—compare side by side.
* **Domain evolution:**

    * **Adding** new rules/constraints is safe—new runs simply include them.
    * **Removing** existing rules/constraints invalidates prior aggregates.

!!! tip "Best Practice"
    If you must modify the domain (especially by dropping elements), **archive** and **clear** previous report data before re‑running to ensure coherent aggregation.

---

## 6. Running the Verification

Once you have your **domain**, **model**, **data**, and **searcher** configured, launch the rule‑consistency check:

```python
from verifia.verification import RuleConsistencyVerifier
from verifia.verification.results import RulesViolationResult

# 1. Prepare your wrapped model and test data
model_wrapper = ...        # e.g. SKLearnModel, CBModel, etc.
test_dataframe = ...       # pandas DataFrame of your test or validation set

# 2. Initialize the verifier with your domain config
verifier = RuleConsistencyVerifier("path/to/domain_rules.yaml")

# 3. Attach the model and data, select search algorithm
verifier = (
    verifier
    .verify(model_wrapper)                 # Model under test
    .on(test_dataframe)                    # Seed data source
    .using("PSO", search_params={          # Choose searcher and its params
        "c_1": 1.0,
        "c_2": 1.0,
        "w_max": 0.9,
        "w_min": 0.4
    })
)

# 4. Run with your verification budget
result: RulesViolationResult = verifier.run(
    pop_size=100,      # candidates per generation
    max_iters=10,      # total number of generations
    orig_seed_ratio=0.5  # optional: use 50% of seed samples
)

# 5. Generate an HTML report
result.save_as_html("model_verification_report.html")

# tip: clean up the result if you remove/edit rules
verifier.clean_results()
```

??? note "Controlling the Seed Sample"

    * `orig_seed_ratio` (`float`, optional)

        * The **fraction** of your original seed rows to inject into each generation.
        * Example: `orig_seed_ratio=0.5` uses **50%** of the DataFrame rows each iteration.

    * `orig_seed_size` (`int`, optional)
        
        * The **absolute number** of seed rows to include.
        * Example: `orig_seed_size=200` always seeds each generation with **200** original samples.

    > **Note:** Only one of `orig_seed_ratio` or `orig_seed_size` should be set. If both are provided, `orig_seed_size` takes precedence. By default, **100%** of the seed data is used (`orig_seed_ratio=1.0`).

??? note "Setting Reporting Config"

    You can customize where VerifIA stores checkpointed reports and how many decimal places are shown in metrics via environment variables:

    ```bash
    export VERIFIA_CHECKPOINTS_DIRPATH="/path/to/checkpoints"
    export VERIFIA_ROUNDING_PRECISION=4
    ```

    * **`VERIFIA_CHECKPOINTS_DIRPATH`**
        * Default: `.verifia/`
        * Directory where run checkpoints and reports are saved.

    * **`VERIFIA_ROUNDING_PRECISION`**
        * Default: `2`
        * Number of decimal places used when rounding and displaying reported metrics.

---

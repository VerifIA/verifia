<div align="center">
  <p>
    <a href="https://semla.polymtl.ca/" target="_blank">
    <img style="border-radius:13px; width:100%; height:auto;" src="https://www.verifia.ca/assets/banner.png" alt="VerifIA x SEMLA banner"></a>
  </p>
</div>

# VerifIA

*VerifIA* is an open-source AI testing framework for **domain‑aware verification** of machine‑learning models during the staging phase—before deployment.

> **Definition:** The <u>staging phase</u> encompasses model training on the training set, hyperparameter tuning on the validation set, and performance evaluation on the test set. During this phase, models must satisfy domain-specific requirements and regulatory standards before advancing to production.

Fundamentally, *VerifIA* automates a structured sequence of verifications to assess model consistency with domain knowledge. At the end of each run, it generates a validation report to:

1. **Inform deployment decisions** by operations teams.
2. **Guide engineers** in debugging and enhancing pipeline robustness.

## Why VerifIA?

Most production AI systems today rely on mature, open‑source frameworks—scikit‑learn, LightGBM, CatBoost, XGBoost, PyTorch, TensorFlow—that keep pace with the latest research and hardware accelerators. While these libraries empower practitioners to build state‑of‑the‑art models on large datasets, many organizations struggle to fully understand or control the resulting systems.  

- **Rush to Adopt:** Businesses integrating AI into existing analytics workflows often end up with only a partial grasp of model behavior.  
- **Regulatory Pressure:** Rising public concern has led to AI‑specific policies and soft laws mandating compliance with human‑centered legal requirements.  
- **Safety‑Critical Caution:** Industries with zero‑tolerance for failure (e.g. aerospace, healthcare) have hesitated to deploy AI without rigorous quality assurance.  

Even rare but catastrophic AI failures—whether due to distributional shifts, spurious correlations, or untested edge cases—are unacceptable in many domains.

*VerifIA* brings **domain‑aware model verification** to your staging pipeline:

1. **Seamless Integration**  
    - **Model frameworks:** [![scikit-learn](https://img.shields.io/badge/scikit--learn-007ACC?logo=scikit-learn&logoColor=white)](https://scikit-learn.org) [![LightGBM](https://img.shields.io/badge/lightgbm-00C1D4?logo=lightgbm&logoColor=white)](https://lightgbm.ai/) [![CatBoost](https://img.shields.io/badge/CatBoost-130C0E?logo=catboost&logoColor=white)](https://catboost.ai/) [![XGBoost](https://img.shields.io/badge/XGBoost-FF6E00?logo=xgboost&logoColor=white)](https://xgboost.ai/) [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
    - **Model registries:** [![MLflow](https://img.shields.io/badge/MLflow-00B0FF?logo=mlflow&logoColor=white)](https://mlflow.org/) [![Comet ML](https://img.shields.io/badge/Comet_ML-1E88E5?logo=comet&logoColor=white)](https://comet.ml/) [![Weights & Biases](https://img.shields.io/badge/Weights_%26_Biases-FF5C8A?logo=wandb&logoColor=white)](https://wandb.ai/)

2. **AI‑Assisted Domain Creation**  
    - Draft your **domain YAML** (variables, constraints, rules) automatically from sample data and documents  
    - Compatible with any LLM via LangChain (e.g. OpenAI, Anthropic, or on‑premise models)  

3. **Systematic Verification**  
    - Run a battery of **rule‑consistency checks** derived from your domain knowledge  
    - Explore out‑of‑sample, in‑domain edge cases via population‑based searchers  

4. **Actionable Reports**  
    - Generate an interactive **validation report** linked back to your staging model  
    - **Operators** decide deployment readiness  
    - **Engineers** gain debugging insights and quality‑improvement guidance  

---

## VerifIA Testing Workflow (Arrange–Act–Assert)

*VerifIA* adopts the well-known **Arrange–Act–Assert** pattern from software testing, adapted for AI model verification:

1. **Arrange:**

      * Define **application-specific expectations**—the domain expert’s assertions about correct model behavior.
      * Structure these assertions into a formal, machine-verifiable format.

2. **Act:**

      * Generate **synthetic input samples** that reflect the original data distribution and explore beyond it.
      * Support targeted generation within specific input subspaces for deeper analysis.

3. **Assert:**

      * Evaluate the model’s responses against the arranged expectations.
      * Quantify compliance levels to identify safe operational domains and uncover inconsistencies.

---

## AI‑Powered Domain Generation

<details>
  <summary>🚀 Preview: VerifIA Domain Spec Generator UI</summary>

  <center>
    <figure>
      <img
        src="https://www.verifia.ca/assets/generation/UI.gif"
        alt="VerifIA Domain Spec Generator UI in action"
        title="Interactive Domain Spec Generator Demo"
        width="85%"
        loading="lazy"
        decoding="async"
      />
      <figcaption><em>Fig. 1:</em> Real‑time domain YAML generation & validation.</figcaption>
    </figure>
  </center>

</details>

Forget manual YAML editing—*VerifIA* can **auto‑draft your entire domain spec** in minutes. 

Using LangChain‑compatible LLMs, it ingests:

- **Your data** (CSV, Parquet, DataFrame)  
- **Your documentation** (PDFs, or vectordb)  
- **Your model card** (YAML file or dict)

It then generates a ready‑to‑use `domain.yaml` that includes:

1. **Variable definitions** (types, real‑world ranges)  
2. **Feasibility constraints** (inter‑feature formulas)  
3. **Behavioral rules** (premises → conclusions)  

!!! tip "Human‑in‑the‑Loop Experience"
    Through the built‑in Gradio interface, you can:
    
    - **Review** the AI‑drafted spec side‑by‑side.  
    - **Edit** any section inline.  
    - **Regenerate** specific parts on demand.  
    - **Validate** against your schema before export. 

👉 Dive into the [AI‑Based Domain Generation Guide](guides/domain-generation/overview.md) for a full walkthrough.  

---

## Quickstart

Install *VerifIA* and run your first verification in minutes:

```bash
pip install verifia
```

```python
from verifia.verification import RuleConsistencyVerifier

# 1. Arrange: load your domain rules
verifier = RuleConsistencyVerifier("domain_rules.yaml")

# 2. Act: attach model (card or dict) + data
report = (
    verifier
      .verify(model_card_fpath_or_dict="model_card.yaml")
      .on(data_fpath="test_data.csv")        # .csv, .json, .xlsx, .parquet, .feather, .pkl
      .using("GA")                           # RS, FFA, MFO, GWO, MVO, PSO, WOA, GA, SSA
      .run(pop_size=50, max_iters=100)       # search budget
)

# 3. Assert: save and inspect your report
report.save_as_html("verification_report.html")
```

👉 [Quickstart guide](/quickstart)

---

## Learn VerifIA

<div class="grid cards" markdown>

-   :material-view-list:{ .lg .middle } __Concepts__

    ---

    Explore VerifIA’s core abstractions—Data, Model, Domain, Searcher, and Run—to understand how they interconnect.

    [:octicons-arrow-right-24: View Concepts](/concepts)

-   :material-school:{ .lg .middle } __Tutorials__

    ---

    Hands‑on walkthroughes that guide you through the core components of VerifIA on simplified examples.

    [:octicons-arrow-right-24: Start Tutorials](/tutos)

-   :material-book-open:{ .lg .middle } __Guides__

    ---

    Deep‑dive guides on domain creation, AI‑Based Domain Generation, and configuration.

    [:octicons-arrow-right-24: Read Guides](/guides/creating-a-domain)

-   :material-rocket-launch:{ .lg .middle } __Use Cases__

    ---

    Discover how teams in finance, healthcare, and beyond can leverage VerifIA to boost model reliability.

    [:octicons-arrow-right-24: Explore Use Cases](/use-cases)

</div>

---

## Contribute

!!! tip "Help us improve *VerifIA*"
    Feel free to:
    
    - [Report a Bug](reporting-a-bug.md)  
    - [Report an Issue](reporting-a-docs-issue.md)  
    - [Request a Change](requesting-a-change.md)  
    - [Make a Pull Request](making-a-pull-request.md)
    
    If you find VerifIA useful, please give us a star on ⭐️[GitHub](https://github.com/VerifIA/verifia)⭐️!


---

## License

*VerifIA* Tool is available under:

| License        | Description                                                                                                                    |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **AGPL‑3.0**   | Community‑driven, OSI‑approved open‑source license                                                                             |
| **Enterprise** | Commercial license for proprietary integration (no AGPL obligations). Mail to [contact@verifia.ca](mailto:contact@verifia.ca). |


We champion open source by returning all improvements back to the community ❤️. Your contributions help advance the field for everyone.
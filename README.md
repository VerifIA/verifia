<p align="center">
  <img alt="VerifIA logo" src="https://www.verifia.ca/assets/logo.png" width="300">
</p>

<h2 align="center" weight='300'>Domain‑Aware Verification Framework for AI Models</h2>

<div align="center">

  [![GitHub release](https://img.shields.io/github/v/release/VerifIA/verifia.svg)](https://github.com/VerifIA/verifia/releases)
  [![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](https://github.com/VerifIA/verifia/blob/main/LICENSE)
  [![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://docs.verifia.ca)

</div>
<h3 align="center">
   <a href="https://docs.verifia.ca"><b>Docs</b></a> &bull;
  <a href="https://www.verifia.ca"><b>Website</b></a>
 </h3>
<br />

---

VerifIA is an open‑source Python library that **automates domain‑aware verification** of machine‑learning models during 
the staging phase—before deployment. 
It generates novel, in‑domain inputs and checks your model against expert‑defined rules, constraints, and specifications, helping you:

- ✅ **Validate** behavioral consistency with domain knowledge  
- 🔍 **Detect** edge‑case failures beyond your labeled data  
- 📊 **Generate** comprehensive HTML reports for decision‑making and debugging

---

## 📖 Try in Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RspOY4zPKWWyXPQ-x0DFpIWh8XyuqI-_?ts=682a261b)

---

## 🚀 Install

```bash
# Core framework
pip install verifia

# Include AI‑Based Domain Generation
pip install verifia[domain]
```

Supports Python 3.10+.

---

## 🤸‍♀️ Quickstart

```python
from verifia.verification import RuleConsistencyVerifier

# 1. Load your domain spec
verifier = RuleConsistencyVerifier("domain_rules.yaml")

# 2. Attach model and data
report = (
    verifier
      .verify(model_card_fpath_or_dict="model_card.yaml")
      .on(data_fpath="test_data.csv")        # .csv, .json, .xlsx, .parquet, .feather, .pkl
      .using("GA")                           # RS, FFA, MFO, GWO, MVO, PSO, WOA, GA, SSA
      .run(pop_size=50, max_iters=100)       # search budget
)

# 3. Save your report
report.save_as_html("verification_report.html")
```

### Quickstart Steps

- **Install**: [install](https://docs.verifia.ca/quickstart/#1-install)  
- **Prepare Your Components** (Domain, Model, Data): [prepare-your-components](https://docs.verifia.ca/concepts/#2-prepare-your-components)  
- **Run Verification**: [run-a-verification](https://docs.verifia.ca/quickstart/#3-run-a-verification)  
- **Inspect Results**: [inspecting-results](https://docs.verifia.ca/quickstart/#4-inspecting-results)  

👉 **Full Quickstart guide**: https://docs.verifia.ca/quickstart

---

## 📚 Feature Spotlight: AI‑Based Domain Generation

Automatically build your domain specification from CSVs, DataFrames, and PDFs using LLM‑powered agents. 
No manual rule‑writing required—point VerifIA at your data and let it generate variables, constraints, and rules for you.

<details style="border:1px solid #ddd; border-radius:8px; background:#fff; padding:1em; margin:1.5em 0; box-shadow:0 2px 8px rgba(0,0,0,0.05);">
  <summary style="font-size:1.15em; font-weight:500; color:#333; cursor:pointer;">
    🎬 Demo: AI‑Based Domain Generation UI
  </summary>
  <div style="text-align:center; margin-top:1em;">
    <a href="https://www.verifia.ca/assets/generation/UI.gif" target="_blank" rel="noopener">
      <div style="display:inline-block; position:relative; overflow:hidden; border-radius:6px;">
        <img
          src="https://www.verifia.ca/assets/generation/UI.gif"
          alt="AI‑Based Domain Generation UI"
          title="Click to view full‑size animation"
          width="80%"
          loading="lazy"
          decoding="async"
          style="display:block;"
        />
        <span style="position:absolute; top:50%; left:50%; transform:translate(-50%, -50%); font-size:3em; color:rgba(255,255,255,0.8); pointer-events:none;">
          ▶️
        </span>
      </div>
    </a>
    <p style="margin:0.75em 0 0; font-size:0.9em; color:#555;">
      <em>Fig.</em> Interactive animated demo—click to open full resolution.
    </p>
  </div>
</details>

### 📖 Learn More

- **Prerequisites & Setup**: [#environment-setup](https://docs.verifia.ca/quickstart)  
- **Prepare Domain Spec**: [#prepare-your-domain](https://docs.verifia.ca/guides/creating-a-domain/)  
- **Run Generation**: [#run-domain-generation](https://docs.verifia.ca/guides/ai-for-domain-generation/)

---

## 🧰 Ecosystem & Integrations

VerifIA works with any model, in any environment and integrates seamlessly with your favorite tools ⤵️

<div align="center">

  [![scikit-learn](https://img.shields.io/badge/scikit--learn-007ACC?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
  [![LightGBM](https://img.shields.io/badge/lightgbm-00C1D4?logo=lightgbm&logoColor=white)](https://lightgbm.ai/)  
  [![CatBoost](https://img.shields.io/badge/CatBoost-130C0E?logo=catboost&logoColor=white)](https://catboost.ai/)
  [![XGBoost](https://img.shields.io/badge/XGBoost-FF6E00?logo=xgboost&logoColor=white)](https://xgboost.ai/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org/)

  [![MLflow](https://img.shields.io/badge/MLflow-00B0FF?logo=mlflow&logoColor=white)](https://mlflow.org/)
  [![Comet ML](https://img.shields.io/badge/Comet_ML-1E88E5?logo=comet&logoColor=white)](https://comet.ml/)
  [![Weights & Biases](https://img.shields.io/badge/Weights_%26_Biases-FF5C8A?logo=wandb&logoColor=white)](https://wandb.ai/)
  [![OpenAI](https://img.shields.io/badge/OpenAI-000000?logo=openai&logoColor=white)](https://openai.com/)
</div>

---

## 📖 More Resources

- **Documentation**: https://docs.verifia.ca
- **Website**: https://verifia.ca 
- **Source Code**: https://github.com/VerifIA/verifia  
- **Contact**: [contact@verifia.ca](mailto:contact@verifia.ca)

---

## 🤝 Contributing

We welcome all contributions! Please read our [CONTRIBUTING.md](https://github.com/VerifIA/verifia/blob/main/CONTRIBUTING.md) to get started.

---

## ⚖️ License

VerifIA is released under the **AGPL‑3.0** License. See [LICENSE](https://github.com/VerifIA/verifia/blob/main/LICENSE) for details.

---

<p align="center">
  Made with ❤️ by the VerifIA contributors.
</p>

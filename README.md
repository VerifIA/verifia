<p align="center">
  <img alt="VerifIA logo" src="https://www.verifia.ca/assets/logo.png" width="300">
</p>

<h2 align="center" weight='300'>Domain‑Aware Verification Framework for AI Models</h2>

<div align="center">

  [![GitHub release](https://img.shields.io/github/v/release/VerifIA/verifia.svg)](https://github.com/VerifIA/verifia/releases)
  [![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](https://github.com/VerifIA/verifia/blob/main/LICENSE)
  [![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://docs.verifia.ai)

</div>
<h3 align="center">
   <a href="https://docs.verifia.ai/index.html"><b>Docs</b></a> &bull;
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

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VerifIA/verifia/blob/main/notebooks/quickstart.ipynb)

---

## 📸 Result Preview [TBD]

[TBD: ADD TEXT HERE]
<p align="center">
  <img src="https://raw.githubusercontent.com/VerifIA/verifia/main/assets/report_preview.png" alt="Verification report preview" width="600">
</p>

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

- **Install**: [docs.verifia.ai/quickstart#install](https://docs.verifia.ai/quickstart#install)  
- **Prepare Your Components** (Domain, Model, Data): [#prepare-your-components](https://docs.verifia.ai/quickstart#prepare-your-components)  
- **Run Verification**: [#run-a-verification](https://docs.verifia.ai/quickstart#run-a-verification)  
- **Inspect Results**: [#inspecting-results](https://docs.verifia.ai/quickstart#inspecting-results)  

👉 **Full Quickstart guide**: https://docs.verifia.ai/quickstart

---

## 📚 Feature Spotlight: AI‑Based Domain Generation

Automatically build your domain specification from CSVs, DataFrames, and PDFs using LLM‑powered agents. 
No manual rule‑writing required—point VerifIA at your data and let it generate variables, constraints, and rules for you.

### 📖 Learn More

- **Prerequisites & Setup**: [#environment-setup](https://docs.verifia.ai/guides/ai-domain-generation/usage#environment-setup)  
- **Prepare Inputs**: [#prepare-your-inputs](https://docs.verifia.ai/guides/ai-domain-generation/usage#prepare-your-inputs)  
- **Run Generation**: [#run-domain-generation](https://docs.verifia.ai/guides/ai-domain-generation/usage#run-domain-generation)  
- **Review & Integrate**: [#review--refine](https://docs.verifia.ai/guides/ai-domain-generation/usage#review--refine) & [#integrate-into-verification-pipeline](https://docs.verifia.ai/guides/ai-domain-generation/usage#integrate-into-verification-pipeline)

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

## 📖 Learn More

- **Documentation**: https://docs.verifia.ai  
- **Website**: https://verifia.ca 
- **Source Code**: https://github.com/VerifIA/verifia  
- **Contact**: [contact@verifia.ai](mailto:contact@verifia.ai)

---

## 🤝 Contributing

We welcome all contributions! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

---

## ⚖️ License

VerifIA is released under the **AGPL‑3.0** License. See [LICENSE](https://github.com/VerifIA/verifia/blob/main/LICENSE) for details.

---

<p align="center">
  Made with ❤️ by the VerifIA contributors.
</p>

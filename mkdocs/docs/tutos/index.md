# Tutorials

Welcome to the *VerifIA* **Tutorials**—two hands‑on guides that walk you through the core components of *VerifIA* on simplified examples. These tutorials will help you:

- **Understand the end‑to‑end workflow** (data → model wrapping → domain spec → searcher selection -> verification run → reporting)  
- **Get comfortable** with the *VerifIA* API before tackling complex, real‑world use cases  
- **Experiment freely**: tweak parameters, swap algorithms, inspect intermediate results  

---

## Tutorial 1: Rule‑Based Dummy Regression Verification

In this tutorial you will learn how to:

1. Generate synthetic regression data with known feature–target relationships  
2. Train a scikit‑learn pipeline  
3. Define domain rules in YAML  
4. Wrap the model and run the `RuleConsistencyVerifier`  
5. Inspect and export verification results  

👉 [Start Tutorial 1](tuto_1.md)

---

## Tutorial 2: Manual vs. AI‑Powered Domain Generation

In this tutorial you will explore two approaches to building your domain configuration on the California Housing dataset:

1. **Manual Domain Dictionary** based on data inspection and expert intuition  
2. **AI‑Powered Generation** using VerifIA’s `DomainGenFlow` (GPT‑assisted)  

You’ll then verify a wrapped model against those rules and export your results.

👉 [Start Tutorial 2](tuto_2.md)

---

> **Next up:**  
> After mastering these tutorials, dive into our [Use Case Gallery](/use-cases) for full‑scale examples across regression and classification that demonstrate VerifIA in realistic settings.  

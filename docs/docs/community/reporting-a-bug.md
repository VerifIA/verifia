# Reporting Bugs

*VerifIA* is an actively maintained open‑source tool. If you believe you’ve found a bug, please help us by opening an issue on our public [issue tracker]. Follow this guide to include the information we need to diagnose and fix problems quickly.

[issue tracker]: https://github.com/verifia/verifia/issues

---

## Before Opening an Issue

With many users and contributors, duplicate or incomplete reports slow us down. Please do the following before filing a new bug report:

### 1. Upgrade to the Latest Version

Bugs are fixed rapidly. Ensure you’re running the [latest release] of VerifIA. If not, update and see if the issue persists.

[latest release]: https://github.com/verifia/verifia/releases

### 2. Search for Existing Reports

- **Documentation**: Search our docs for related configuration or usage notes.
- **Issues**: Browse the issue tracker to see if someone else has reported the same problem.

Keep track of your search terms and any relevant links—you’ll include them in your report.

---

## Issue Template

When you’re sure the bug isn’t already known or fixed, open a new issue using this template. Fill in each section as completely as possible.

### 1. Title

**Short & descriptive**, e.g. “`verify()` crashes when domain YAML has empty `constraints` block.”

### 2. Context *(optional)*

Briefly explain your environment or use case. For example:

> Running VerifIA v1.2.0 on Ubuntu 22.04 with Python 3.10, using a custom domain definition for robotic joint constraints.

### 3. Bug Description

- **What happened?** A concise summary of the unexpected behavior.
- **Why it’s a bug**: Explain why this isn’t intended VerifIA behavior.

### 4. Related Links

List any docs pages, errors, StackOverflow questions, or existing issues you’ve found while investigating.

### 5. Reproduction

Provide a minimal example (YAML, Python snippet, test data) that triggers the bug. Paste it directly or attach a `.zip` (≤1 MB).

### 6. Steps to Reproduce

Numbered steps showing exactly how to run your minimal example and observe the bug.

1. `pip install verifia==0.1.0`  
2. Create `domain.yaml` with…  
3. Run:
   ```python
      verifier = ...
   ```

### 7. Environment *(optional)*

* **OS**: e.g. Ubuntu 22.04, Windows 10
* **Python**: e.g. 3.9.7
* **VerifIA**: e.g. v0.1.0
* **Additional packages**: if relevant

---

## Checklist

Before submitting, please confirm:

* [x] I’m on the latest VerifIA version
* [x] I searched docs and existing issues
* [x] My report includes a clear title and description
* [x] I provided a minimal reproduction
* [x] I listed exact steps to reproduce
* [x] I attached relevant logs or error messages

---

_Thank you for helping make *VerifIA* more reliable !_


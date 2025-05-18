"""
Constants for the VerifIA Domain Spec Generator UI.

This module centralizes all UI-related constants, including theme configuration,
URLs, placeholder texts, labels, button texts, and layout settings.
"""
from pathlib import Path
from gradio import themes

# -----------------------------
# UI Theme
# -----------------------------
VERIFIA_THEME = themes.Soft()

# -----------------------------
# URLs
# -----------------------------
LOGO_URL: str = "https://www.verifia.ca/assets/logo.png"
GITHUB_REPO_URL: str = "https://github.com/VerifIA/verifia"

# -----------------------------
# Application Titles
# -----------------------------
APP_TITLE: str = "VerifIA Domain Spec Generator"
HEADER_TITLE: str = "Domain Spec Generator"

# -----------------------------
# YAML Display Settings
# -----------------------------
YAML_PLACEHOLDER: str = "# Generated YAML will appear here"
YAML_LINES: int = 25
YAML_MAX_LINES: int = 25

# -----------------------------
# Instructions Panel
# -----------------------------
INSTRUCTIONS_TITLE: str = "✏️ Instructions"
INSTRUCTIONS_PLACEHOLDER: str = "e.g. 'Please remove the age constraint'"
INSTRUCTIONS_LINES: int = 2

# -----------------------------
# Labels
# -----------------------------
LABEL_YAML: str = "📜 YAML Specification"
LABEL_VALIDATION: str = "🔍 Validation Output"
VALIDATION_LINES: int = 4

# -----------------------------
# Button Texts
# -----------------------------
BUTTON_START: str = "🚀 Start Generation"
BUTTON_VALIDATE: str = "✅ Validate"
BUTTON_REGENERATE: str = "🔄 Regenerate"
BUTTON_FINISH: str = "🎉 Finish"

# -----------------------------
# Paths
# -----------------------------
CSS_PATH: Path = Path(__file__).parent / "assets" / "style.css"

# -----------------------------
# Footer HTML
# -----------------------------
FOOTER_HTML: str = (
    "<p style='text-align:center; width:100%;'>"
    "Built with ❤️ by the VerifIA team &mdash; "
    f"⭐ Enjoying VerifIA? <a href='{GITHUB_REPO_URL}'>Star us on GitHub!</a> ⭐"
    "</p>"
)
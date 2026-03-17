"""Sphinx configuration for agri-harvest documentation."""

project = "agri-harvest"
copyright = "2025, AGRI-HARVEST Team"
author = "AGRI-HARVEST Team"
release = "2.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "alabaster"
html_static_path = ["_static"]

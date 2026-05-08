"""Sphinx configuration for the langchain-goodmem documentation site."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "examples"))

project = "langchain-goodmem"
author = "langchain-goodmem contributors"
copyright = "2026, langchain-goodmem contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_autodoc_typehints",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
root_doc = "index"
exclude_patterns = ["_build", "build", "Thumbs.db", ".DS_Store"]
templates_path: list[str] = []
autosummary_generate = False
autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
typehints_use_rtype = False
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False
add_module_names = False

html_title = "langchain-goodmem"
html_baseurl = "https://z8kh8E6t-rEv62qT7.github.io/langchain_goodmem/"

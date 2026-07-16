"""Sphinx configuration for the LightningMedSeg3D documentation."""
from __future__ import annotations

import os
import sys

# Make the package importable for autodoc without installing it.
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information ------------------------------------------------------
project = "LightningMedSeg3D"
author = "Marcos Fdez-Gonzalez and contributors"
copyright = "2026, LightningMedSeg3D authors"
release = "1.0.0"
version = "1.0.0"

# -- General configuration ----------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}

# Heavy runtime dependencies are mocked so the API can be documented on
# Read the Docs without installing PyTorch/MONAI.
autodoc_mock_imports = [
    "torch", "lightning", "pytorch_lightning", "monai", "nibabel", "numpy",
    "scipy", "sklearn", "einops", "tensorboard", "jsonargparse", "pandas",
    "matplotlib", "tqdm", "yaml",
]
autodoc_default_options = {"members": True, "undoc-members": True, "show-inheritance": True}
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "lightning": ("https://lightning.ai/docs/pytorch/stable", None),
    "monai": ("https://docs.monai.io/en/stable", None),
}

# -- HTML output (Furo theme) -------------------------------------------------
html_theme = "furo"
html_title = "LightningMedSeg3D"
html_static_path = ["_static"]
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.png"
html_show_sourcelink = False
pygments_style = "friendly"
pygments_dark_style = "monokai"

html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#2C5F8A",
        "color-brand-content": "#2E7D5B",
    },
    "dark_css_variables": {
        "color-brand-primary": "#7FB2D9",
        "color-brand-content": "#5FBE93",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/Removirt/LightningMedSeg3D",
            "html": "",
            "class": "fa-brands fa-github",
        },
    ],
}

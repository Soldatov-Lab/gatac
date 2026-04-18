"""Sphinx configuration for GATAC documentation."""

import sys
from datetime import datetime
from pathlib import Path

# Add the project root to the path so autodoc can find the package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Pre-import packages that must NOT be loaded inside sphinx's mock context.
# sphinx mocks cupy/cupyx; anndata's compat layer tries to import them at load
# time, and the mock interceptor causes a deadlock on first import inside the
# mock context.  Pre-importing here ensures they are already in sys.modules
# before any mock context is entered.
# Inspired by scverse/rapids-singlecell docs/conf.py.
import anndata  # noqa: E402, F401
import scanpy   # noqa: E402, F401


# -- Project information -----------------------------------------------------
project = "GATAC"
copyright = f"{datetime.now().year}, GATAC contributors"
author = "GATAC contributors"

# The full version, including alpha/beta/rc tags
try:
    from gatac import __version__
    release = __version__
    version = ".".join(release.split(".")[:2])
except ImportError:
    release = "0.1.0"
    version = "0.1"

# -- General configuration ---------------------------------------------------
extensions = [
    # Core Sphinx
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    # Typing
    "sphinx_autodoc_typehints",
    # Notebooks + Markdown (myst_nb already loads myst_parser)
    "myst_nb",
    # UX
    "sphinx_copybutton",
    "sphinx_design",
]

# MyST extensions
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
    "substitution",
]

myst_substitutions = {
    "version": release,
}

# Notebook execution
nb_execution_mode = "off"  # Notebooks pre-executed; set to "auto" for live builds
nb_execution_timeout = 600

# Napoleon settings (NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
napoleon_preprocess_types = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autosummary_generate = True

# Mock GPU / optional imports that are not available in the docs environment
autodoc_mock_imports = [
    "cudf",
    "cupy",
    "cuml",
    "cudf_polars",
    "cupyx",
    "rmm",
    "pyfaidx",
    "MOODS",
]

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "polars": ("https://docs.pola.rs/api/python/stable/", None),
}
intersphinx_timeout = 5  # seconds; avoids hanging on blocked network in CI/HPC

# Copy button settings
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# With SOURCEDIR set to the repo root (see docs/Makefile), Sphinx needs to
# know which document is the root and which paths to exclude from scanning.
root_doc = "docs/index"

templates_path = ["_templates"]
exclude_patterns = [
    "docs/_build",
    "**/Thumbs.db",
    "**/.DS_Store",
    "**/.ipynb_checkpoints",
    # Non-docs repo content
    "gatac/**",
    "tests/**",
    "data/**",
    "refs/**",
    ".git/**",
    ".venv/**",
    ".pytest_cache/**",
    "__pycache__/**",
    "*.log",
    # Top-level files not part of the docs
    "*.py",
    "*.toml",
    "*.lock",
    "AGENTS.md",
    "GEMINI.md",
    "README.md",
]

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "GATAC"

html_theme_options = {
    "logo": {
        "text": "GATAC",
    },
    "github_url": "https://github.com/Soldatov-Lab/gatac",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "navbar_persistent": ["search-button"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "show_toc_level": 2,
    "navigation_depth": 3,
    "collapse_navigation": True,
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    "announcement": None,
}

html_sidebars = {
    "index": [],
    "installation": [],
    "contributing": [],
    "changelog": [],
    "api/**": ["sidebar-nav-bs"],
    "cli/**": ["sidebar-nav-bs"],
    "tutorials/**": ["sidebar-nav-bs"],
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_context = {
    "default_mode": "light",
}

# -- Source suffix -----------------------------------------------------------
# myst_nb owns both .md and .ipynb; use "myst-nb" so Sphinx can find the parser.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

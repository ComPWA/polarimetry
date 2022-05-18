import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.abspath("."))
from _relink_references import relink_references


def get_execution_mode() -> str:
    if "EXECUTE_NB" in os.environ:
        print("\033[93;1mWill run Jupyter notebooks!\033[0m")
        return "cache"
    return "off"


def generate_api() -> None:
    shutil.rmtree("api", ignore_errors=True)
    subprocess.call(
        " ".join(
            [
                "sphinx-apidoc",
                f"../src/polarization/",
                f"../src/polarization/version.py",
                "-o api/",
                "--force",
                "--no-toc",
                "--separate",
                "--templatedir _templates",
            ]
        ),
        shell=True,
    )


generate_api()
relink_references()


add_module_names = False
author = "Mikhail Mikhasenko, Remco de Boer"
autodoc_default_options = {
    "exclude-members": ", ".join(
        [
            "default_assumptions",
            "doit",
            "evaluate",
            "is_commutative",
            "is_extended_real",
        ]
    ),
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints_format = "short"
codeautolink_concat_default = True
copyright = "2022"
exclude_patterns = [
    "**.ipynb_checkpoints",
    ".DS_Store",
    "Thumbs.db",
    "_build",
]
extensions = [
    "myst_nb",
    "sphinx.ext.intersphinx",
    "sphinx_book_theme",
    "sphinx_codeautolink",
    "sphinx_copybutton",
    "sphinx_panels",
    "sphinx_togglebutton",
]
html_sourcelink_suffix = ""
html_theme = "sphinx_book_theme"
html_theme_options = {
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
    },
    "path_to_docs": "docs",
    "repository_url": "https://github.com/redeboer/polarization-sensitivity",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
}
html_title = "Polarization sensitivity"
intersphinx_mapping = {
    "IPython": ("https://ipython.readthedocs.io/en/stable", None),
    "ampform": (f"https://ampform.readthedocs.io/en/stable", None),
    "attrs": (f"https://www.attrs.org/en/stable", None),
    "ipywidgets": (f"https://ipywidgets.readthedocs.io/en/stable", None),
    "matplotlib": (f"https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
    "sympy": ("https://docs.sympy.org/latest", None),
    "tensorwaves": (f"https://tensorwaves.readthedocs.io/en/stable", None),
}
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "substitution",
]
nb_execution_allow_errors = False
nb_execution_mode = get_execution_mode()
nb_execution_timeout = -1
nb_output_stderr = "show"
nb_render_markdown_format = "myst"
numfig = True
panels_add_bootstrap_css = False
pygments_style = "sphinx"
use_multitoc_numbering = True

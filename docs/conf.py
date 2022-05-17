import os


def get_execution_mode() -> str:
    if "EXECUTE_NB" in os.environ:
        print("\033[93;1mWill run Jupyter notebooks!\033[0m")
        return "cache"
    return "off"


author = "Mikhail Mikhasenko, Remco de Boer"
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
    "sphinx_copybutton",
    "sphinx_external_toc",
    "sphinx_panels",
    "sphinx_togglebutton",
]
external_toc_path = "_toc.yml"
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
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "substitution",
]
nb_execution_allow_errors = False
nb_execution_excludepatterns = [
    "polarization.ipynb",
]
nb_execution_mode = get_execution_mode()
nb_execution_timeout = -1
nb_output_stderr = "show"
nb_render_markdown_format = "myst"
numfig = True
panels_add_bootstrap_css = False
pygments_style = "sphinx"
use_multitoc_numbering = True

import os
import shutil
import subprocess
import sys
from textwrap import dedent

sys.path.insert(0, os.path.abspath("."))
from _relink_references import relink_references


def download_figure_1() -> str:
    files = [
        "figure1.svg",
        "figure1-inset.svg",
    ]
    for file in files:
        if not os.path.exists(file):
            print_missing_file_warning(file)
            return ""
    return f"""
    ```{{only}} html
    High-resolution image can be downloaded here: {{download}}`{files[0]}` / {{download}}`{files[1]}`
    ```
    """.strip()


def download_figures_2_and_3() -> str:
    files = [
        "figure2.svg",
        "figure2-inset.svg",
        "figure3a.svg",
        "figure3a-inset.svg",
        "figure3b.svg",
        "figure3b-inset.svg",
    ]
    for file in files:
        if not os.path.exists(file):
            print_missing_file_warning(file)
            return ""

    src = f"""
    ```{{only}} html
    **Figures 2 and 3** for the paper can be downloaded here:

    - {{download}}`{files[0]}` / {{download}}`{files[1]}`
    - {{download}}`{files[2]}` / {{download}}`{files[3]}`
    - {{download}}`{files[4]}` / {{download}}`{files[5]}`
    ```
    """
    return dedent(src).strip()


def download_intensity_distribution() -> str:
    filename = "intensity-distribution.png"
    if not os.path.exists(filename):
        print_missing_file_warning(filename)
        return ""
    src = f"""
    ```{{only}} html
    High-resolution image can be downloaded here: {{download}}`{filename}`
    ```
    """
    return dedent(src).strip()


def execute_pluto_notebooks() -> None:
    if "EXECUTE_PLUTO" not in os.environ:
        return
    if shutil.which("julia") is None:
        raise ValueError(
            "Julia is not installed. Please download it at`"
            " https://julialang.org/downloads"
        )
    result = subprocess.call(
        "julia --project=. ./exportnotebooks.jl",
        cwd="../julia",
        shell=True,
    )
    if result != 0:
        raise ValueError("Failed to execute pluto notebooks")


def get_execution_mode() -> str:
    if "FORCE_EXECUTE_NB" in os.environ:
        print("\033[93;1mWill run ALL Jupyter notebooks!\033[0m")
        return "force"
    if "EXECUTE_NB" in os.environ:
        print("\033[93;1mWill run Jupyter notebooks with cache\033[0m")
        return "cache"
    return "off"


def get_link_to_julia_pages() -> str:
    julia_landing_page = "./_static/julia/index.html"
    if os.path.exists(julia_landing_page):
        src = f"""
        :::{{tip}}
        Several cross-checks with Julia can be found [here]({julia_landing_page}).
        :::
        """
        return dedent(src)
    return ""


def get_nb_remove_code_source():
    if "latex" in sys.argv[2]:
        print(f"\033[91;1mCell input will not be rendered\033[0m")
        return True
    return False


def generate_api() -> None:
    shutil.rmtree("api", ignore_errors=True)
    subprocess.call(
        " ".join(
            [
                "sphinx-apidoc",
                f"../src/polarimetry/",
                f"../src/polarimetry/version.py",
                "-o api/",
                "--force",
                "--no-toc",
                "--separate",
                "--templatedir _templates",
            ]
        ),
        shell=True,
    )


def get_link_to_single_pdf() -> str:
    build_file = "_build/latex/python.pdf"
    embedded_file = "_static/polarimetry.pdf"
    if os.path.exists(build_file):
        shutil.copy(build_file, embedded_file)
    if os.path.exists(embedded_file):
        src = f"""
        ::::{{only}} html
        :::{{tip}}
        This webpage can be downloaded as a **single PDF file** [here]({embedded_file}).
        :::
        ::::
        """
        return dedent(src)
    print(f"\033[91;1mSingle PDF has not yet been built.\033[0m")
    return ""


def print_missing_file_warning(filename: str) -> None:
    print(f"\033[93;1m{filename} not found, so cannot create download links\033[0m")


execute_pluto_notebooks()
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
autodoc_type_aliases = {
    "OuterStates": "polarimetry.decay.OuterStates",
}
autodoc_typehints_format = "short"
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2
codeautolink_concat_default = True
copyright = "2022"
default_role = "py:obj"
exclude_patterns = [
    "**.ipynb_checkpoints",
    ".DS_Store",
    "Thumbs.db",
    "_build",
]
extensions = [
    "myst_nb",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_book_theme",
    "sphinx_codeautolink",
    "sphinx_copybutton",
    "sphinx_panels",
    "sphinx_togglebutton",
    "sphinxcontrib.inkscapeconverter",
]
html_sourcelink_suffix = ""
html_static_path = ["_static"]
html_theme = "sphinx_book_theme"
html_theme_options = {
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
    },
    "path_to_docs": "docs",
    "repository_url": "https://github.com/ComPWA/polarimetry",
    "repository_branch": "main",
    "show_navbar_depth": 2,
    "show_toc_level": 2,
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
}
html_title = "Polarimetry Λc → p K π"
intersphinx_mapping = {
    "IPython": ("https://ipython.readthedocs.io/en/stable", None),
    "ampform": (f"https://ampform.readthedocs.io/en/stable", None),
    "attrs": ("https://www.attrs.org/en/stable", None),
    "ipywidgets": ("https://ipywidgets.readthedocs.io/en/stable", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sympy": ("https://docs.sympy.org/latest", None),
    "tensorwaves": ("https://tensorwaves.readthedocs.io/en/stable", None),
}
latex_elements = {
    "papersize": "a4paper",
    "preamble": R"""
\usepackage{bookmark}
\usepackage[Latin,Greek]{ucharclasses}
\usepackage{unicode-math}
\hypersetup{
    pdfencoding=auto,
    psdextra
}

\bookmarksetup{
  numbered,
  addtohook={%
    \ifnum\bookmarkget{level}>1 %
      \bookmarksetup{numbered=false}%
    \fi
  },
}
""",
}
latex_appendices = [
    "appendix/dynamics",
    "appendix/angles",
    "appendix/phase-space",
    "appendix/alignment",
    "appendix/benchmark",
    "appendix/serialization",
]
latex_engine = "xelatex"  # https://tex.stackexchange.com/a/570691
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "substitution",
]
myst_render_markdown_format = "myst"
myst_substitutions = {
    "DOWNLOAD_SINGLE_PDF": get_link_to_single_pdf(),
    "LINK_TO_JULIA_PAGES": get_link_to_julia_pages(),
    "download_figure_1": download_figure_1(),
    "download_figures_2_and_3": download_figures_2_and_3(),
    "download_intensity_distribution": download_intensity_distribution(),
}
nb_execution_allow_errors = False
nb_execution_mode = get_execution_mode()
nb_execution_show_tb = True
nb_execution_timeout = -1
nb_output_stderr = "show"
nb_render_markdown_format = "myst"
nb_remove_code_source = get_nb_remove_code_source()
nitpicky = False
nitpick_ignore_regex = [
    ("py:class", "KeyType"),
    ("py:class", "NewValueType"),
    ("py:class", "OldValueType"),
]
numfig = True
panels_add_bootstrap_css = False
primary_domain = "py"
pygments_style = "sphinx"
suppress_warnings = [
    # skipping unknown output mime type: application/json
    # https://gitlab.cern.ch/polarimetry/Lc2pKpi/-/jobs/24273321#L2123
    "mystnb.unknown_mime_type",
]
use_multitoc_numbering = True
viewcode_follow_imported_members = True

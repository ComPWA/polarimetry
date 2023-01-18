import os
import shutil
import subprocess
import sys
from datetime import datetime
from textwrap import dedent, indent

if sys.version_info < (3, 8):
    from importlib_metadata import PackageNotFoundError
    from importlib_metadata import version as get_package_version
else:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as get_package_version

sys.path.insert(0, os.path.abspath("extensions"))


def download_paper_figures() -> str:
    files = [
        "_static/images/total-polarimetry-field-watermark.svg",
        "_static/images/polarimetry-field-L1520-unaligned-watermark.svg",
        "_static/images/polarimetry-field-L1520-aligned-watermark.svg",
        "_static/images/polarimetry-field-norm-uncertainties-watermark.png",
    ]
    for file in files:
        if not os.path.exists(file):
            print_missing_file_warning(file)
            return ""
    list_of_figures = indent("\n".join(_to_download_link(f) for f in files), "    - ")
    src = f"""
    ::::{{only}} html
    :::{{tip}}
    Figures for the paper can be downloaded here:
    {list_of_figures.strip()}

    All other exported figures can be [here](./_static/images/).
    :::
    ::::
    """
    return dedent(src).strip()


def download_intensity_distribution() -> str:
    filename = "_static/images/intensity-distribution.png"
    if not os.path.exists(filename):
        print_missing_file_warning(filename)
        return ""
    src = f"""
    ```{{only}} html
    High-resolution image can be downloaded here: {_to_download_link(filename)}
    ```
    """
    return dedent(src).strip()


def _to_download_link(path: str) -> str:
    basename = os.path.basename(path)
    return f"[`{basename}`]({path})"


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


def get_timestamp() -> str:
    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")


def get_version() -> str:
    try:
        return get_package_version("polarimetry")
    except PackageNotFoundError:
        return ""


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
    build_file = "_build/latex/polarimetry.pdf"
    embedded_file = "_static/polarimetry.pdf"
    if os.path.exists(build_file):
        shutil.copy(build_file, embedded_file)
    if os.path.exists(embedded_file):
        src = f"""
        ::::{{only}} html
        :::{{button-link}} {embedded_file}
        :color: primary
        :shadow:
        Download this website as a **report**
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


add_module_names = False
author = "Mikhail Mikhasenko, Remco de Boer, Miriam Fritsch"
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
bibtex_bibfiles = [
    "_static/references.bib",
]
codeautolink_concat_default = True
copyright = "2022"
default_role = "py:obj"
exclude_patterns = [
    "**.ipynb_checkpoints",
    ".DS_Store",
    "Thumbs.db",
    "_build",
    "_static/export/README.md",
]
extensions = [
    "myst_nb",
    "relink_references",
    "sphinx_reredirects",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_book_theme",
    "sphinx_codeautolink",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.inkscapeconverter",
    "support_bibtex_math",
    "unsrt_et_al",
]
html_css_files = [
    "custom.css",
]
html_js_files = [
    # https://github.com/requirejs/requirejs/tags
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js",
]
html_sourcelink_suffix = ""
html_static_path = ["_static"]
html_theme = "sphinx_book_theme"
html_theme_options = {
    "announcement": (
        "⚠️This webpage has been frozen in the state that was used for <a"
        ' href="https://arxiv.org/abs/2301.07010v1">arXiv:2301.07010v1</a>⚠️<br>Visit'
        ' <a href="https://github.com/ComPWA/polarimetry">github.com/ComPWA/polarimetry</a>'
        " for the latest version of the codebase!"
    ),
    "extra_navbar": f"<p>Version {get_version()} ({get_timestamp()})</p>",
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
html_title = "Polarimetry Λ<sub>c</sub> → p K π"
intersphinx_mapping = {
    "IPython": ("https://ipython.readthedocs.io/en/stable", None),
    "ampform": ("https://ampform.readthedocs.io/en/stable", None),
    "attrs": ("https://www.attrs.org/en/stable", None),
    "iminuit": ("https://iminuit.readthedocs.io/en/stable", None),
    "ipywidgets": ("https://ipywidgets.readthedocs.io/en/stable", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "plotly": ("https://plotly.com/python-api-reference", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sympy": ("https://docs.sympy.org/latest", None),
    "tensorwaves": ("https://tensorwaves.readthedocs.io/en/stable", None),
}
latex_documents = [
    # https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-latex_documents
    (
        "index",
        "polarimetry.tex",
        R"""
        $\Lambda_c$ polarimetry using the dominant hadronic mode ― supplemental material
        """.strip(),
        author,
        "manual",
        False,
    ),
]
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
    "releasename": f"{get_version()} ({get_timestamp()})",
}
latex_engine = "xelatex"  # https://tex.stackexchange.com/a/570691
latex_show_pagerefs = True
linkcheck_ignore = [
    "https://arxiv.org/pdf/2208.03262.pdf",
    "https://arxiv.org/pdf/hep-ex/0510019.pdf",
    "https://github.com/ComPWA/polarimetry",
    "https://journals.aps.org/prd/pdf/10.1103/PhysRevD.101.034033",
]
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "html_image",
    "substitution",
]
myst_render_markdown_format = "myst"
myst_substitutions = {
    "DOWNLOAD_SINGLE_PDF": get_link_to_single_pdf(),
    "LINK_TO_JULIA_PAGES": get_link_to_julia_pages(),
    "DOWNLOAD_PAPER_FIGURES": download_paper_figures(),
    "DOWNLOAD_INTENSITY_DISTRIBUTION": download_intensity_distribution(),
}
relink_ref_types = {
    "jax.numpy.ndarray": "obj",
    "polarimetry.decay.OuterStates": "obj",
    "polarimetry.lhcb.ParameterType": "obj",
    "tensorwaves.interface.DataSample": "obj",
    "tensorwaves.interface.Function": "obj",
    "tensorwaves.interface.ParameterValue": "obj",
    "tensorwaves.interface.ParametrizedFunction": "obj",
}
relink_targets = {
    "Axes": "matplotlib.axes.Axes",
    "DataSample": "tensorwaves.interface.DataSample",
    "Function": "tensorwaves.interface.Function",
    "Literal[(-1, 1)]": "typing.Literal",
    "Literal[- 1, 1]": "typing.Literal",
    "Literal[-1, 1]": "typing.Literal",
    "OuterStates": "polarimetry.decay.OuterStates",
    "ParameterType": "polarimetry.lhcb.ParameterType",
    "ParameterValue": "tensorwaves.interface.ParameterValue",
    "ParametrizedFunction": "tensorwaves.interface.ParametrizedFunction",
    "Path": "pathlib.Path",
    "Pattern": "typing.Pattern",
    "PoolSum": "ampform.sympy.PoolSum",
    "PositionalArgumentFunction": "tensorwaves.function.PositionalArgumentFunction",
    "QuadContourSet": "matplotlib.contour.QuadContourSet",
    "UnevaluatedExpression": "ampform.sympy.UnevaluatedExpression",
    "implement_doit_method": "ampform.sympy.implement_doit_method",
    "jnp.ndarray": "jax.numpy.ndarray",
    "polarimetry.lhcb._T": "typing.TypeVar",
    "sp.Expr": "sympy.core.expr.Expr",
    "sp.Indexed": "sympy.tensor.indexed.Indexed",
    "sp.Mul": "sympy.core.mul.Mul",
    "sp.Rational": "sympy.core.numbers.Rational",
    "sp.Symbol": "sympy.core.symbol.Symbol",
    "sp.acos": "sympy.functions.elementary.trigonometric.acos",
    "typing.Literal[-1, 1]": "typing.Literal",
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
primary_domain = "py"
pygments_style = "sphinx"
redirects = {
    "appendix/polarization-fit": "../zz.polarization-fit.html",
}
suppress_warnings = [
    "mystnb.mime_priority",  # plotly figures in LaTeX build
    # https://github.com/executablebooks/MyST-NB/blob/4dcf7c5/docs/conf.py#L46-L47
    "mystnb.unknown_mime_type",
]
use_multitoc_numbering = True
version = get_version()
viewcode_follow_imported_members = True

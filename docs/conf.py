from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from textwrap import dedent, indent

import requests
from attrs import define, field
from sphinx_api_relink.helpers import (
    get_package_version,
    pin,
    pin_minor,
    set_intersphinx_version_remapping,
)

sys.path.insert(0, os.path.abspath("extensions"))


def download_paper_figures() -> str:
    figures = {
        "2": "_static/images/total-polarimetry-field-watermark.svg",
        "3a": "_static/images/polarimetry-field-L1520-unaligned-watermark.svg",
        "3b": "_static/images/polarimetry-field-L1520-aligned-watermark.svg",
        "4": "_static/images/polarimetry-field-norm-uncertainties-watermark.png",
    }
    for path in figures.values():
        if not os.path.exists(path):
            MISSING_FILES.add(path)
            return ""
    list_of_figures = indent(
        "\n".join(
            f":Figure {name}: {_to_download_link(path)}"
            for name, path in figures.items()
        ),
        4 * " ",
    )
    src = f"""
    ::::{{only}} html
    :::{{tip}}
    Figures for the paper can be downloaded here:
    {list_of_figures.strip()}

    All other exported figures can be found [here](./_static/images/).
    :::
    ::::
    """
    return dedent(src).strip()


def download_intensity_distribution() -> str:
    filename = "_static/images/intensity-distribution.png"
    if not os.path.exists(filename):
        MISSING_FILES.add(filename)
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
        msg = "Julia is not installed. Please download it at https://julialang.org/downloads"
        raise ValueError(msg)
    result = subprocess.call(
        "julia --project=. ./exportnotebooks.jl",
        cwd="../julia",
        shell=True,
    )
    if result != 0:
        msg = "Failed to execute pluto notebooks"
        raise ValueError(msg)


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
        print("\033[91;1mCell input will not be rendered\033[0m")
        return True
    return False


def get_figure_link(
    rel_path: str, cwd: str | None = None, full_width: bool = False
) -> str:
    abs_path = Path(rel_path)
    if cwd is not None:
        abs_path = Path(cwd) / rel_path
    if not abs_path.exists():
        MISSING_FILES.add(abs_path)
        return ""
    if full_width:
        src = f"""
        ```{{container}} full-width
        {get_figure_link(rel_path, cwd, full_width=False)}
        ```
        """
        return dedent(src).strip()
    return f"![]({rel_path}"


def get_polarimeter_figures_side_by_side() -> str:
    paths = (
        "_static/images/polarimetry-field-K-contours-title-watermark-inset.svg",
        "_static/images/polarimetry-field-L-contours-title-watermark-inset.svg",
        "_static/images/polarimetry-field-D-contours-title-watermark-inset.svg",
    )
    if any(not os.path.exists(p) for p in paths):
        return ""
    src = "```{container} full-width\n"
    for p in paths:
        src += f'<img src="{p}" width="33%">'
    src += "```\n"
    return src


def get_polarimeter_chain_figures() -> str:
    paths = (
        "_static/images/polarimetry-K-chains.svg",
        "_static/images/polarimetry-L-chains.svg",
        "_static/images/polarimetry-D-chains.svg",
    )
    if any(not os.path.exists(p) for p in paths):
        return ""
    src = "```{container} full-width\n"
    for p in paths:
        src += get_figure_link(p) + "\n"
    src += "```\n"
    return src


def get_timestamp() -> str:
    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")


def get_link_to_single_pdf() -> str:
    build_file = "_build/latex/polarimetry.pdf"
    embedded_file = "_static/polarimetry.pdf"
    if os.path.exists(build_file):
        shutil.copy(build_file, embedded_file)
    if os.path.exists(embedded_file):
        src = f"""
        :::{{grid-item-card}} {{octicon}}`download` Download this website as a single PDF file
        :columns: 12
        :link: {embedded_file}
        :::
        """
        return dedent(src)
    print("\033[91;1mSingle PDF has not yet been built.\033[0m")
    return ""


def get_scipy_url() -> str:
    url = f"https://docs.scipy.org/doc/scipy-{pin('scipy')}/"
    r = requests.get(url)
    if r.status_code != 200:
        return "https://docs.scipy.org/doc/scipy"
    return url


@define
class MissingFileCollector:
    paths: list[Path] = field(factory=list)

    def add(self, path: str | Path) -> None:
        path = Path(path)
        rel_path = path.resolve().relative_to(Path.cwd())
        self.paths.append(rel_path)

    def print(self) -> None:
        if len(self.paths) == 0:
            return
        print("\033[93;1mFollowing files are missing and cannot be embedded:\033[0m")
        for path in sorted(self.paths):
            print(f"  \033[93;1m {path} \033[0m")


execute_pluto_notebooks()
set_intersphinx_version_remapping({
    "ipywidgets": {
        "8.1.1": "8.1.2",
    },
    "matplotlib": {
        "3.9.1.post1": "3.9.1",
    },
})

MISSING_FILES = MissingFileCollector()

BRANCH = "main"
PACKAGE = "polarimetry-lc2pkpi"
PACKAGE_NAME = "polarimetry"
ORGANIZATION = "ComPWA"
REPO_NAME = "polarimetry"

add_module_names = False
api_github_repo = f"{ORGANIZATION}/{REPO_NAME}"
api_target_substitutions = {
    "AmplitudeModel": "ampform_dpd.AmplitudeModel",
    "Artist": "matplotlib.artist.Artist",
    "Axes": "matplotlib.axes.Axes",
    "BreitWignerMinL": "ampform_dpd.dynamics.BreitWignerMinL",
    "BuggBreitWigner": "ampform_dpd.dynamics.BuggBreitWigner",
    "DalitzPlotDecompositionBuilder": "ampform_dpd.DalitzPlotDecompositionBuilder",
    "DataSample": ("obj", "tensorwaves.interface.DataSample"),
    "FlattéSWave": "ampform_dpd.dynamics.FlattéSWave",
    "Function": ("obj", "tensorwaves.interface.Function"),
    "implement_doit_method": "ampform.sympy.implement_doit_method",
    "LineCollection": "matplotlib.collections.LineCollection",
    "LineshapeName": ("obj", "polarimetry.lhcb.LineshapeName"),
    "Literal[- 1, 1]": "typing.Literal",
    "Literal[-1, 1]": "typing.Literal",
    "Literal[(-1, 1)]": "typing.Literal",
    "ModelName": ("obj", "polarimetry.lhcb.ModelName"),
    "np.ndarray": ("obj", "numpy.ndarray"),
    "OuterStates": ("obj", "polarimetry.decay.OuterStates"),
    "ParameterType": ("obj", "polarimetry.lhcb.ParameterType"),
    "ParameterValue": ("obj", "tensorwaves.interface.ParameterValue"),
    "ParametrizedFunction": ("obj", "tensorwaves.interface.ParametrizedFunction"),
    "Particle": "ampform_dpd.decay.Particle",
    "Path": "pathlib.Path",
    "Pattern": "typing.Pattern",
    "polarimetry.lhcb._T": "typing.TypeVar",
    "PoolSum": "ampform.sympy.PoolSum",
    "PositionalArgumentFunction": "tensorwaves.function.PositionalArgumentFunction",
    "QuadContourSet": "matplotlib.contour.QuadContourSet",
    "ResonanceName": ("obj", "polarimetry.lhcb.ResonanceName"),
    "sp.acos": "sympy.functions.elementary.trigonometric.acos",
    "sp.Expr": "sympy.core.expr.Expr",
    "sp.Indexed": "sympy.tensor.indexed.Indexed",
    "sp.Mul": "sympy.core.mul.Mul",
    "sp.Rational": "sympy.core.numbers.Rational",
    "sp.Symbol": "sympy.core.symbol.Symbol",
    "SympyDataTransformer": "tensorwaves.data.transform.SympyDataTransformer",
    "ThreeBodyDecay": "ampform_dpd.decay.ThreeBodyDecay",
    "ThreeBodyDecayChain": "ampform_dpd.decay.ThreeBodyDecayChain",
    "typing.Literal[-1, 1]": "typing.Literal",
    "UnevaluatedExpression": "ampform.sympy.UnevaluatedExpression",
}
api_target_types = {
    "jax.numpy.ndarray": "obj",
}
author = "Mikhail Mikhasenko, Remco de Boer, Miriam Fritsch"
autodoc_default_options = {
    "exclude-members": ", ".join([  # noqa: FLY002
        "default_assumptions",
        "doit",
        "evaluate",
        "is_commutative",
        "is_extended_real",
    ]),
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
bibtex_default_style = "unsrt_et_al"
codeautolink_concat_default = True
copyright = "2023"
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
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_api_relink",
    "sphinx_book_theme",
    "sphinx_codeautolink",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_pybtex_etal_style",
    "sphinx_reredirects",
    "sphinx_togglebutton",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.inkscapeconverter",
    "support_bibtex_math",
]
generate_apidoc_package_path = f"../src/{PACKAGE_NAME}"
html_css_files = [
    "custom.css",
]
html_js_files = [
    # https://github.com/requirejs/requirejs/tags
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js",
]
html_last_updated_fmt = "%-d %B %Y"
html_logo = "_static/lhcb-logo.svg"
html_sourcelink_suffix = ""
html_static_path = ["_static"]
html_theme = "sphinx_book_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "Common Partial Wave Analysis",
            "url": "https://compwa.github.io",
            "icon": "https://compwa.github.io/_static/favicon.ico",
            "type": "url",
        },
        {
            "name": "GitHub",
            "url": f"https://github.com/{ORGANIZATION}/{REPO_NAME}",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": f"https://pypi.org/project/{PACKAGE}",
            "icon": "fa-brands fa-python",
        },
        {
            "name": "Launch on Binder",
            "url": f"https://mybinder.org/v2/gh/{ORGANIZATION}/{REPO_NAME}/{BRANCH}?filepath=docs",
            "icon": "https://mybinder.readthedocs.io/en/latest/_static/favicon.png",
            "type": "url",
        },
        {
            "name": "Launch on Colaboratory",
            "url": f"https://colab.research.google.com/github/{ORGANIZATION}/{REPO_NAME}/blob/{BRANCH}",
            "icon": "https://avatars.githubusercontent.com/u/33467679?s=100",
            "type": "url",
        },
    ],
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com",
        "deepnote_url": "https://deepnote.com",
        "notebook_interface": "jupyterlab",
    },
    "logo": {"text": "Λ<sub>c</sub>⁺&nbsp;→&nbsp;p&nbsp;K⁻&nbsp;π⁺"},
    "path_to_docs": "docs",
    "repository_branch": BRANCH,
    "repository_url": f"https://github.com/{ORGANIZATION}/{REPO_NAME}",
    "show_navbar_depth": 1,
    "show_toc_level": 2,
    "use_download_button": False,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_source_button": True,
}
html_title = "Λ<sub>c</sub> → p K π polarimetry"
intersphinx_mapping = {
    "IPython": (f"https://ipython.readthedocs.io/en/{pin('IPython')}", None),
    "ampform-dpd": ("https://compwa.github.io/ampform-dpd", None),
    "ampform": (f"https://ampform.readthedocs.io/en/{pin('ampform')}", None),
    "attrs": (f"https://www.attrs.org/en/{pin('attrs')}", None),
    "iminuit": ("https://iminuit.readthedocs.io/en/stable", None),
    "ipywidgets": (f"https://ipywidgets.readthedocs.io/en/{pin('ipywidgets')}", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "matplotlib": (f"https://matplotlib.org/{pin('matplotlib')}", None),
    "numpy": (f"https://numpy.org/doc/{pin_minor('numpy')}", None),
    "plotly": ("https://plotly.com/python-api-reference", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": (get_scipy_url(), None),
    "sympy": ("https://docs.sympy.org/latest", None),
    "tensorwaves": (f"https://tensorwaves.rtfd.io/en/{pin('tensorwaves')}", None),
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
    "releasename": f"{get_package_version(PACKAGE)} ({get_timestamp()})",
}
latex_engine = "xelatex"  # https://tex.stackexchange.com/a/570691
latex_show_pagerefs = True
linkcheck_ignore = [
    "https://arxiv.org/pdf/2208.03262.pdf",
    "https://arxiv.org/pdf/hep-ex/0510019.pdf",
    "https://journals.aps.org/prd/pdf/10.1103/PhysRevD.101.034033",
    "https://www.bookfinder.com",
]
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "fieldlist",
    "html_image",
    "substitution",
]
myst_render_markdown_format = "myst"
myst_substitutions = {
    "DOWNLOAD_INTENSITY_DISTRIBUTION": download_intensity_distribution(),
    "DOWNLOAD_PAPER_FIGURES": download_paper_figures(),
    "DOWNLOAD_SINGLE_PDF": get_link_to_single_pdf(),
    "FIG_ALPHA_Z_STAT": get_figure_link(
        "_images/alpha-z-per-resonance-statistical.svg"
    ),
    "FIG_ALPHA_Z_SYST": get_figure_link(
        "_images/alpha-z-per-resonance-systematics.svg"
    ),
    "FIG_ALPHA_XZ_STAT": get_figure_link("_images/alpha-xz-statistics.svg"),
    "FIG_ALPHA_XZ_SYST": get_figure_link("_images/alpha-xz-systematics.svg"),
    "FIG_ALPHA_XZ_STAT_PLOTLY": get_figure_link(
        "_images/alpha-xz-statistics-plotly.svg"
    ),
    "FIG_ALPHA_XZ_SYST_PLOTLY": get_figure_link(
        "_images/alpha-xz-systematics-plotly.svg"
    ),
    "FIG_CORRELATION_STAT": get_figure_link("_images/correlation-statistics.svg"),
    "FIG_CORRELATION_SYST": get_figure_link("_images/correlation-systematics.svg"),
    "FIG_CORRELATION_MAT": get_figure_link("_images/correlation-matrices.svg"),
    "FIG_INTENSITY": get_figure_link("_images/intensity-distributions-1D.svg"),
    "FIG_INTENSITY_SIG1": get_figure_link("_images/intensity-distributions-sigma1.svg"),
    "FIG_INTENSITY_SIG2": get_figure_link("_images/intensity-distributions-sigma2.svg"),
    "FIG_INTENSITY_SIG3": get_figure_link("_images/intensity-distributions-sigma3.svg"),
    "FIG_PHASE_SPACE": get_figure_link(
        "../_images/phase-space-boundary.svg",
        cwd="appendix",
    ),
    "FIG_POLARIMETER_CHAIN": get_polarimeter_chain_figures(),
    "FIG_POLARIMETER_SUBSYSTEM": get_figure_link(
        "_static/images/polarimetry-per-subsystem.svg"
    ),
    "FIG_POLARIZATION_SYST": get_figure_link(
        "_static/images/polarization-distribution-systematics.svg"
    ),
    "FIG_POLARIMETER_TOTAL": get_polarimeter_figures_side_by_side(),
    "FIG_RATE_MATRIX": get_figure_link("_images/rate-matrix.svg"),
    "FIG_RATE_MATRIX_SUB": get_figure_link("_images/rate-matrix-sub-region.svg"),
    "FIG_SUB_REGIONS": get_figure_link("_images/sub-regions.svg"),
    "LINK_TO_JULIA_PAGES": get_link_to_julia_pages(),
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
version = get_package_version(PACKAGE)

MISSING_FILES.print()

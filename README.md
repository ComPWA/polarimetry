# Aligned polarimetry field of the Λc → p π K decay

[![10.1007/JHEP07(2023)228](<https://zenodo.org/badge/doi/10.1007/JHEP07(2023)228.svg>)](<https://doi.org/10.1007/JHEP07(2023)228>)
[![10.5281/zenodo.7544989](https://zenodo.org/badge/doi/10.5281/zenodo.7544989.svg)](https://doi.org/10.5281/zenodo.7544989)
[![GPLv3+ license](https://img.shields.io/badge/License-GPLv3+-blue.svg)](https://www.gnu.org/licenses/gpl-3.0-standalone.html)

[![code style: prettier](https://img.shields.io/badge/code_style-prettier-ff69b4.svg?style=flat-square)](https://github.com/prettier/prettier)
[![Spelling checked](https://img.shields.io/badge/cspell-checked-brightgreen.svg)](https://github.com/streetsidesoftware/cspell/tree/master/packages/cspell)
[![PyPI package](https://badge.fury.io/py/polarimetry-lc2pki.svg)](https://pypi.org/project/polarimetry-lc2pki)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/polarimetry-lc2pki)](https://pypi.org/project/polarimetry-lc2pki)

This repository a symbolic amplitude model for the decay $\Lambda^+_c \to p \pi^+ K^-$ that is aligned with [Dalitz-plot decomposition](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.034033) and computes an align polarimeter vector field $\vec\alpha$. Helicity couplings and other parameter values are taken from a recent study by the LHCb Collaboration[^1] and its [supplementary material](https://cds.cern.ch/record/2824328/files).

<!-- cspell:ignore semileptonic -->

[^1]: Amplitude analysis of the $\Lambda^+_c \to p K^- \pi^+$ decay and $\Lambda^+_c$ baryon polarization measurement in semileptonic beauty hadron decays (2022) [[link]](https://inspirehep.net/literature/2132745)

## Installation

It's recommended to develop this code base with [VSCode](https://code.visualstudio.com) and install the developer environment with Conda:

```shell
conda env create
conda activate polarimetry
```

Style checks are enforced with [Pre-commit](https://pre-commit.com). To activate for each commit, run:

```shell
pre-commit install
```

For more information about local Python set-up, see [here](https://compwa-org.readthedocs.io/develop.html#local-set-up).

This repository also contains Julia source code and Pluto notebooks. Julia can be downloaded [here](https://julialang.org/downloads). You then have to activate and instantiated the Julia environment provided in the [`julia`](./julia) folder. This can be done as follows from the root directory:

```shell
julia --project=./julia -e 'import Pkg; Pkg.instantiate()'
```

### Documentation dependencies

To build the documentation, you need to install LaTeX and some additional fonts. In Ubuntu, this can be done with:

```shell
sudo apt-get install -y cm-super dvipng texlive-latex-extra
```

In addition, for [building the documentation as a single PDF file](#building-the-documentation), you need to install XeTeX:

```shell
sudo apt-get install -y inkscape latexmk make texlive-fonts-extra texlive-xetex xindy
```

<!-- cspell:ignore xetex -->

## Building the documentation

Having [installed the Python environment](#installation), you can build the documentation with:[^2]

```shell
tox -e docnb
```

This will run all Jupyter notebooks and convert the output to static webpages (view the output under `docs/_build/html/index.html`). Running all notebooks from scratch (without any available cache) should take **around one hour**.

If you have installed Julia and instantiated the Julia environment, you can embed the [Pluto notebooks](./julia/notebooks) as static pages in the documentation with:

```shell
EXECUTE_PLUTO=YES tox -e docnb
```

or, alternatively, by executing _all_ Jupyter and Pluto notebooks (ignoring any existing caches):

```shell
EXECUTE_PLUTO=YES tox -e docnb-force
```

The [above commands](#building-the-documentation) result in a static HTML webpage. It's also possible to render the notebook as a single PDF file. This can be done as follows:

```shell
tox -e pdf
```

Just as above, cell output can be rendered by setting the `EXECUTE_NB` variable to some value:

```shell
EXECUTE_NB=YES tox -e pdf
```

[^2]:
    It's also possible have a look at the documentation _without_ cell output (just as a check for the links). This can be done with:

    ```shell
    tox -e doc
    ```

# Aligned polarimetry field of the Λc → p π K decay

[![10.1007/JHEP07(2023)228](<https://zenodo.org/badge/doi/10.1007/JHEP07(2023)228.svg>)](<https://doi.org/10.1007/JHEP07(2023)228>)
[![10.5281/zenodo.7544989](https://zenodo.org/badge/doi/10.5281/zenodo.7544989.svg)](https://doi.org/10.5281/zenodo.7544989)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![code style: prettier](https://img.shields.io/badge/code_style-prettier-ff69b4.svg?style=flat-square)](https://github.com/prettier/prettier)
[![Spelling checked](https://img.shields.io/badge/cspell-checked-brightgreen.svg)](https://github.com/streetsidesoftware/cspell/tree/main/packages/cspell)
[![PyPI package](https://badge.fury.io/py/polarimetry-lc2pkpi.svg)](https://pypi.org/project/polarimetry-lc2pkpi)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/polarimetry-lc2pkpi)](https://pypi.org/project/polarimetry-lc2pkpi)
[![Binder](https://static.mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ComPWA/polarimetry/stable?urlpath=lab)

This repository contains the source code for “<span></span>$\Lambda_c^+$ polarimetry using the dominant hadronic mode” (2023) by the LHCb Collaboration ([10.1007/JHEP07(2023)228](<https://doi.org/10.1007/JHEP07(2023)228>)). It uses [`ampform-dpd`](https://github.com/ComPWA/ampform-dpd) to formulate symbolic amplitude models for the decay $\Lambda^+_c \to p \pi^+ K^-$ that are aligned with [Dalitz-plot decomposition](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.034033). The aligned amplitudes are used to compute polarimeter vector field $\vec\alpha$ over the Dalitz plane. Helicity couplings and other parameter values are taken from a recent study by the LHCb Collaboration[^1] and its [supplementary material](https://cds.cern.ch/record/2824328/files).

<!-- cspell:ignore semileptonic -->

[^1]: Amplitude analysis of the $\Lambda^+_c \to p K^- \pi^+$ decay and $\Lambda^+_c$ baryon polarization measurement in semileptonic beauty hadron decays (2022) [[link]](https://inspirehep.net/literature/2132745)

## Installation

All the dependencies required to work on this Repository are managed by [Pixi](https://pixi.sh). If you have installed Pixi, you can simply start developing by activating the environment with:

```shell
pixi shell
```

This will install [`uv`](https://docs.astral.sh/uv) for managing Python and its dependencies as well as Julia. Then it activates the environment and you can start developing.

Common checks are defined under [`pixi.toml`](./pixi.toml) and can be listed with

```shell
pixi task list
```

For example, you build the documentation with executed notebooks with

```shell
pixi run docnb
```

Style checks are enforced with [Pre-commit](https://pre-commit.com). You should install this system-wide once, for instance through [`uv`](https://docs.astral.sh/uv):

```shell
uv tool install --with pre-commit-uv pre-commit
```

You can then install the Git hooks for this repository, so that the checks are run automatically before each commit:

```shell
pre-commit install
```

> [!TIP]
> For more information about the local Python developer environment, see [here](https://compwa.github.io/develop#local-set-up).

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
pixi run docnb
```

This will run all Jupyter notebooks and convert the output to static webpages (view the output under `docs/_build/html/index.html`). Running all notebooks from scratch (without any available cache) should take **around one hour**.

If you have installed Julia and instantiated the Julia environment, you can embed the [Pluto notebooks](./julia/notebooks) as static pages in the documentation with:

```shell
pixi run docnb-pluto
```

or, alternatively, by executing _all_ Jupyter and Pluto notebooks (ignoring any existing caches):

```shell
pixi run docnb-pluto-force
```

The [above commands](#building-the-documentation) result in a static HTML webpage. It's also possible to render the notebook as a single PDF file. This can be done as follows:

```shell
pixi run pdf
```

Just as above, cell output can be rendered by appending `nb`:

```shell
pixi run pdfnb
```

[^2]:
    It's also possible have a look at the documentation _without_ cell output (just as a check for the links). This can be done with:

    ```shell
    pixi run doc
    ```

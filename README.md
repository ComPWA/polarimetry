# Polarization sensitivity in Λc → pπK

<!-- cspell:ignore semileptonic -->

[![GPLv3+ license](https://img.shields.io/badge/License-GPLv3+-blue.svg)](https://www.gnu.org/licenses/gpl-3.0-standalone.html)

This repository a symbolic amplitude model for the decay $\Lambda^+_c \to p \pi^+ K^-$ that is aligned with [Dalitz-plot decomposition](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.034033) and computes an align polarimeter vector field $\vec\alpha$. Helicity couplings and other parameter values are taken from [_Amplitude analysis of the $\Lambda^+_c \to p K^- \pi^+$ decay and $\Lambda^+_c$ baryon polarization measurement in semileptonic beauty hadron decays_](https://inspirehep.net/literature/2132745) (2022) by the LHCb Collaboration and its [supplementary material](https://cds.cern.ch/record/2824328/files).

## Installation

It's recommended to develop this code base with [VSCode](https://code.visualstudio.com) and install the developer environment with Conda:

```shell
conda env create
conda activate polarization
```

Style checks are enforced with [Pre-commit](https://pre-commit.com). To activate for each commit, run:

```shell
pre-commit install
```

For more information about local Python set-up, see [here](https://compwa-org.readthedocs.io/develop.html#local-set-up).

### Documentation dependencies

To build the documentation, you need to install LaTeX and some additional fonts. In Ubuntu, this can be done with:

<!-- cspell:ignore dvipng texlive -->

```shell
sudo apt install -y cm-super dvipng texlive-latex-extra
```

# Polarization sensitivity in Λc → pπK

<!-- cspell:ignore Flatte modelparameters modelstudies -->

[![GPLv3+ license](https://img.shields.io/badge/License-GPLv3+-blue.svg)](https://www.gnu.org/licenses/gpl-3.0-standalone.html)

This repository originates from [ComPWA/compwa-org#129](https://github.com/ComPWA/compwa-org/pull/129). It formulates a symbolic amplitude model for the decay Λc → pπK that is aligned with [Dalitz-plot decomposition](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.034033) and computes

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

## Data

The parameters of the LHCb fit are stored in the json files in the folder [`data/`](data/)

### Content of the [`data/isobars.json`](data/isobars.json)

The file provides the description of the characteristics and parameters of the resonances in the decay chains.
The resonances are named by `L(XXXX)`, `D(XXXX)`, and `K(XXXX)` for the Lambda**, Delta**, and K\*\* states.
The data structure has the fields `jp`, `mass`, `width`, and `lineshape`.

There are three lineshape types used:

- "BreitWignerMinL": the standard parametrization with isobar spectator orbital momentum set to its minimal value
- "BuggBreitWignerMinL": the mass-dependent width incorporates Adler zero and exponential form factor
- "Flatte1405": the mass-dependent width includes two terms, pK and Sigma pi with the same Gamma0

For most of the resonances, the width field gives a fixed value.
However, for a few, an interval is provided. In that case, the width was a parameter of the fit.
Its exact value is to be found in the list of parameters

### Content of the [`data/modelparameters.json`](data/modelparameters.json)

The fit results is stored in the `first(file["modelstudies"])`. The dictionary contains the list of all floating parameters.
`ArF(XXXX)N` and `AiF(XXXX)N` are the real and imaginary part of the coupling `K^{Lc->Fx}`, where

- the `F` stands for `D`,`L`, or `K`.
- the `N` numbers the helicity indices with the following mapping

```
# Lambda**
L(XXXX)1 -> K^{Lc->Lambda pi}_{+1/2,0}
L(XXXX)2 -> K^{Lc->Lambda pi}_{-1/2,0}
# Delta**
D(XXXX)1 -> K^{Lc->Delta K}_{+1/2,0}
D(XXXX)2 -> K^{Lc->Delta K}_{-1/2,0}
# scalar K: K(700) and K(1430)
K(XXXX)1 -> K^{Lc->Delta K}_{0,-1/2}
K(XXXX)2 -> K^{Lc->Delta K}_{0,+1/2}
# other K: K(892)
K(XXXX)1 -> K^{Lc->Lpi}_{ +,1/2}
K(XXXX)2 -> K^{Lc->Lpi}_{-1,1/2}
K(XXXX)3 -> K^{Lc->Lpi}_{+1,-1/2}
K(XXXX)4 -> K^{Lc->Lpi}_{ 0,-1/2}
```

The default-fit result is stored in as the first item, `first(file["modelstudies"])`.

# Data

<!-- cspell:ignore modelstudies -->

The parameters of the LHCb fit are stored in the json files in the folder [`data/`](./)

## Content of the [`particle-definitions/.yaml`](./particle-definitions.yaml)

The file provides the description of the characteristics and parameters of the resonances in the decay chains. The resonances are named by `L(XXXX)`, `D(XXXX)`, and `K(XXXX)` for the $\Lambda^{**}$, $\Delta^{**}$, and $K^{**}$ states. The data structure has the fields `jp`, `mass`, `width`, and `lineshape`.

There are three lineshape types:

- `"BreitWignerMinL"`: the standard parametrization with isobar spectator orbital momentum set to its minimal value
- `"BuggBreitWignerMinL"`: the mass-dependent width incorporates Adler zero and exponential form factor
- `"Flatte1405"`: the mass-dependent width includes two terms, pK and Sigma pi with the same Gamma0

For most of the resonances, the width field gives a fixed value. However, for a few, an interval is provided. In that case, the width was a parameter of the fit. Its exact value is to be found in the list of parameters

## Content of [`model-definitions.yaml`](./model-definitions.yaml)

The fit results is stored in the `first(file["modelstudies"])`. The dictionary contains the list of all floating parameters. `ArF(XXXX)N` and `AiF(XXXX)N` are the real and imaginary part of the coupling `K^{Lc->Fx}`, where

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
K(XXXX)1 -> K^{Lc->Lpi}_{ 0,+1/2}
K(XXXX)2 -> K^{Lc->Lpi}_{-1,+1/2}
K(XXXX)3 -> K^{Lc->Lpi}_{+1,-1/2}
K(XXXX)4 -> K^{Lc->Lpi}_{ 0,-1/2}
```

The default-fit result is stored in as the first item, `first(file["modelstudies"])`.

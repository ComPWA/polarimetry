---
myst:
  all_links_external: True
---

# Polarimetry in Λ<sub>c</sub>⁺&nbsp;→&nbsp;p&nbsp;K⁻&nbsp;π⁺

:::{title} Welcome
:::

[![10.48550/arXiv.2301.07010](https://zenodo.org/badge/doi/10.48550/arXiv.2301.07010.svg)](https://doi.org/10.48550/arXiv.2301.07010)
[![10.5281/zenodo.7544989](https://zenodo.org/badge/doi/10.5281/zenodo.7544989.svg)](https://doi.org/10.5281/zenodo.7544989)

<!-- cspell:disable -->

:::{card} $\Lambda^+_\mathrm{c}$ polarimetry using the dominant hadronic mode
:link: https://doi.org/10.1007/JHEP07(2023)228

The polarimeter vector field for multibody decays of a spin-half baryon is introduced as a generalisation of the baryon asymmetry parameters. Using a recent amplitude analysis of the $\Lambda^+_\mathrm{c} \to p K^- \pi^+$ decay performed at the LHCb experiment, we compute the distribution of the kinematic-dependent polarimeter vector for this process in the space of Mandelstam variables to express the polarised decay rate in a model-agnostic form. The obtained representation can facilitate polarisation measurements of the $\Lambda^+_\mathrm{c}$ baryon and eases inclusion of the $\Lambda^+_\mathrm{c} \to p K^- \pi^+$ decay mode in hadronic amplitude analyses.

<!-- cspell:enable -->

:::

:::::{only} html
::::{grid} 1 2 2 2
:margin: 4 4 0 0
:gutter: 1

:::{grid-item-card} {material-regular}`functions` Symbolic expressions
:link: amplitude-model
:link-type: doc

Compute the amplitude model over large data samples with symbolic expressions.
:::

:::{grid-item-card} {octicon}`file-code` JSON grids
:link: exported-distributions
:link-type: ref

Reuse the computed polarimeter field in any amplitude analysis involving $\Lambda_\mathrm{c}^+$.
:::

:::{grid-item-card} {material-regular}`ads_click` Inspect interactively
:link: appendix/widget
:link-type: doc

Investigate how parameters in the amplitude model affect the polarimeter field.
:::

:::{grid-item-card} {octicon}`book` Compute polarization
:link: zz.polarization-fit
:link-type: doc

Learn how to determine the polarization vector using the polarimeter field.
:::

{{ DOWNLOAD_SINGLE_PDF }}
::::
:::::

This website shows all analysis results that led to the publication of [LHCb-PAPER-2022-044](https://cds.cern.ch/record/2838694). More information on this publication can be found on the following pages:

- Publication on JHEP: [J. High Energ. Phys. 2023, 228 (2023)](<https://doi.org/10.1007/JHEP07(2023)228>)
  <!-- cspell:ignore Energ -->
- Publication on arXiv: [arXiv:2301.07010](https://arxiv.org/abs/2301.07010)
- Record on CDS: [cds.cern.ch/record/2838694](https://cds.cern.ch/record/2838694)
- Record for the source code on Zenodo: [10.5281/zenodo.7544989](https://doi.org/10.5281/zenodo.7544989)
- Archived documentation on GitLab Pages: [lc2pkpi-polarimetry.docs.cern.ch](https://lc2pkpi-polarimetry.docs.cern.ch)
- Archived repository on CERN GitLab: [gitlab.cern.ch/polarimetry/Lc2pKpi](https://gitlab.cern.ch/polarimetry/Lc2pKpi)
- Active repository on GitHub containing discussions: [github.com/ComPWA/polarimetry](https://github.com/ComPWA/polarimetry)

:::{admonition} Behind SSO login (LHCb members only)
:class: toggle

- LHCb TWiki page: [twiki.cern.ch/twiki/bin/viewauth/LHCbPhysics/PolarimetryLc2pKpi](https://twiki.cern.ch/twiki/bin/viewauth/LHCbPhysics/PolarimetryLc2pKpi)
- Charm WG meeting: [indico.cern.ch/event/1187317](https://indico.cern.ch/event/1187317)
- RC approval presentation: [indico.cern.ch/event/1213570](https://indico.cern.ch/event/1213570)
- Silent approval to submit: [indico.cern.ch/event/1242323](https://indico.cern.ch/event/1242323)

:::

<!-- cspell:ignore lc2pkpi -->

::::{only} latex
:::{note}
This document is a PDF rendering of the supplemental material hosted behind SSO-login on [lc2pkpi‑polarimetry.docs.cern.ch](https://lc2pkpi-polarimetry.docs.cern.ch). Go to this webpage for a more extensive and interactive experience.
:::
::::

[![PyPI package](https://badge.fury.io/py/polarimetry-lc2pki.svg)](https://pypi.org/project/polarimetry-lc2pki)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/polarimetry-lc2pki)](https://pypi.org/project/polarimetry-lc2pki)

Each of the pages contain code examples for how to reproduce the results with the Python package hosted at [github.com/ComPWA/polarimetry](https://github.com/ComPWA/polarimetry). However, to quickly get import the model in another package, it is possible to install the package from PyPI:

```bash
pip install lc2pkpi-polarimetry
```

:::{autolink-concat}
:::

Each of the models can then simply be imported as

```python
import polarimetry

model = polarimetry.published_model()
```

<!-- cspell:ignore maxdepth -->

```{toctree}
---
caption: Table of contents
maxdepth: 2
numbered:
---
amplitude-model
cross-check
intensity
polarimetry
uncertainties
resonance-polarimetry
appendix
references
API <api/polarimetry>
```

:::{only} html

```{toctree}
---
caption: External links
hidden:
---
JHEP <https://doi.org/10.1007/JHEP07(2023)228>
arXiv:2301.07010 <https://arxiv.org/abs/2301.07010>
ComPWA <https://compwa-org.readthedocs.io>
GitHub repository <https://github.com/ComPWA/polarimetry>
CERN GitLab (archived) <https://gitlab.cern.ch/polarimetry/Lc2pKpi>
```

:::

{{ LINK_TO_JULIA_PAGES }}

{{ DOWNLOAD_PAPER_FIGURES }}

::::{dropdown} Notebook execution times
:::{nb-exec-table}
:::
::::

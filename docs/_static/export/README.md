# Computed polarimetry fields

This folder contains the aligned polarimetry vector field $\vec\alpha$ computed over a 100x100 grid over the Dalitz plane for the decay $\Lambda_c^+ \to p\pi^+K^-$.

The computed fields are made available in JSON format with the following keys:

- **`metadata`**: description of the model used to compute the field. Contains:
  - `model description`: indicates whether the `"Default amplitude model"` was used or one of the alternative models
  - `parameters`: a dictionary of parameter names with their values (can be float, integer, or complex).
  - `reference subsystem`: the subsystem ID (including LaTeX description) of the subsystem that was used to align the amplitudes with [Dalitz Plot Decomposition](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.0340330).
- **`m^2_Kpi`**: an array of 100 values for $\sigma_1 = m^2(K^-,\pi^+)$ that span the $x$-axis of the Dalitz grid.
- **`m^2_pK`**: an array of 100 values for $\sigma_2 = m^2(p,K^-)$ that span the $y$-axis of the Dalitz grid.
- **`alpha_x`**, **`alpha_y`**, **`alpha_z`**: computed values for $\alpha_x$, $\alpha_y$, $\alpha_z$ over a 100x100 grid arrays each.[^1]
- **`intensity`**: computed intensity on each Dalitz grid point.[^1]

[^1]: Grid points that lie outside the phase phase space are given as Not-a-Number (NaN).

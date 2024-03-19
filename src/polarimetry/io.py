"""Import-output of the polarimeter field and improvements to printing in notebooks."""

from __future__ import annotations

import json
import logging

import jax
import jax.numpy as jnp
import sympy as sp
from ampform_dpd.io import aslatex  # pyright:ignore[reportPrivateImportUsage]
from IPython.core.display import Math
from IPython.display import display


def display_latex(obj) -> None:
    latex = aslatex(obj)
    display(Math(latex))


def display_doit(expr: sp.Expr, deep=False, terms_per_line: int | None = None) -> None:
    if terms_per_line is None:
        latex = aslatex({expr: expr.doit(deep=deep)})
    else:
        latex = sp.multiline_latex(
            lhs=expr,
            rhs=expr.doit(deep=deep),
            terms_per_line=terms_per_line,
            environment="eqnarray",
        )
    display(Math(latex))


def mute_jax_warnings() -> None:
    jax_logger = logging.getLogger("absl")
    jax_logger = logging.getLogger("jax._src.lib.xla_bridge")
    jax_logger.setLevel(logging.ERROR)


def export_polarimetry_field(  # noqa: PLR0917
    sigma1: jax.Array,
    sigma2: jax.Array,
    alpha_x: jax.Array,
    alpha_y: jax.Array,
    alpha_z: jax.Array,
    intensity: jax.Array,
    filename: str,
    metadata: dict | None = None,
) -> None:
    if len(sigma1.shape) != 1:
        msg = f"sigma1 must be a 1D array, got {sigma1.shape}"
        raise ValueError(msg)
    if len(sigma2.shape) != 1:
        msg = f"sigma2 must be a 1D array, got {sigma2.shape}"
        raise ValueError(msg)
    expected_shape: tuple[int, int] = (*sigma1.shape, *sigma2.shape)
    for array in [alpha_x, alpha_y, alpha_z, intensity]:
        if array.shape != expected_shape:
            msg = f"Expected shape {expected_shape}, got {array.shape}"
            raise ValueError(msg)
    json_data = {
        "m^2_Kpi": sigma1.tolist(),
        "m^2_pK": sigma2.tolist(),
        "alpha_x": alpha_x.tolist(),
        "alpha_y": alpha_y.tolist(),
        "alpha_z": alpha_z.tolist(),
        "intensity": intensity.tolist(),
    }
    if metadata is not None:
        json_data = {
            "metadata": metadata,
            **json_data,
        }
    with open(filename, "w") as f:
        json.dump(json_data, f, separators=(",", ":"))


def import_polarimetry_field(filename: str, steps: int = 1) -> dict[str, jax.Array]:
    with open(filename) as f:
        json_data: dict = json.load(f)
    return {
        "m^2_Kpi": jnp.array(json_data["m^2_Kpi"])[::steps],
        "m^2_pK": jnp.array(json_data["m^2_pK"])[::steps],
        "alpha_x": jnp.array(json_data["alpha_x"])[::steps, ::steps],
        "alpha_y": jnp.array(json_data["alpha_y"])[::steps, ::steps],
        "alpha_z": jnp.array(json_data["alpha_z"])[::steps, ::steps],
        "intensity": jnp.array(json_data["intensity"])[::steps, ::steps],
    }

"""Import-output of the polarimeter field and improvements to printing in notebooks."""

from __future__ import annotations

import json
import logging
import warnings
from typing import Any

import jax.numpy as jnp
import sympy as sp
from ampform_dpd.io import aslatex
from IPython.core.display import Math
from IPython.display import display


def display_latex(obj, *, wrap: bool = False) -> None:
    if wrap:
        latex = aslatex(obj, terms_per_line=1)
    else:
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


def mute_ampform_warnings() -> None:
    logging.getLogger("ampform.sympy._cache").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning, module="ampform_dpd.decay")


def mute_jax_warnings() -> None:
    logging.getLogger("absl").setLevel(logging.ERROR)
    logging.getLogger("jax._src.lib.xla_bridge").setLevel(logging.ERROR)
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)


def export_polarimetry_field(  # noqa: PLR0917
    sigma1: jnp.ndarray,
    sigma2: jnp.ndarray,
    alpha_x: jnp.ndarray,
    alpha_y: jnp.ndarray,
    alpha_z: jnp.ndarray,
    intensity: jnp.ndarray,
    filename: str,
    metadata: dict | None = None,
) -> None:
    if len(sigma1.shape) != 1:
        msg = f"sigma1 must be a 1D array, got {sigma1.shape}"
        raise ValueError(msg)
    if len(sigma2.shape) != 1:
        msg = f"sigma2 must be a 1D array, got {sigma2.shape}"
        raise ValueError(msg)
    expected_shape: tuple[int, int] = (*sigma1.shape, *sigma2.shape)  # ty:ignore[invalid-assignment]
    for array in [alpha_x, alpha_y, alpha_z, intensity]:
        if array.shape != expected_shape:
            msg = f"Expected shape {expected_shape}, got {array.shape}"
            raise ValueError(msg)
    json_data: dict[str, Any] = {
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


def import_polarimetry_field(filename: str, steps: int = 1) -> dict[str, jnp.ndarray]:
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

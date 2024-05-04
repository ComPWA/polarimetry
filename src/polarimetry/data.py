"""Helper functions for importing data and creating data transformers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import sympy as sp
from ampform.kinematics.phasespace import is_within_phasespace
from ampform_dpd import create_mass_symbol_mapping, formulate_third_mandelstam
from ampform_dpd.decay import ThreeBodyDecay
from tensorwaves.data import IntensityDistributionGenerator, NumpyDomainGenerator
from tensorwaves.data.rng import NumpyUniformRNG
from tensorwaves.data.transform import SympyDataTransformer
from tensorwaves.function.sympy import create_function

if TYPE_CHECKING:
    from ampform_dpd import AmplitudeModel
    from ampform_dpd.decay import FinalStateID, ThreeBodyDecay
    from tensorwaves.function import PositionalArgumentFunction
    from tensorwaves.interface import DataSample


def create_data_transformer(
    model: AmplitudeModel, backend: str = "jax"
) -> SympyDataTransformer:
    kinematic_variables = {
        symbol: expression.doit().subs(model.parameter_defaults)
        for symbol, expression in model.variables.items()
    }
    identity_mapping = {s: s for s in sp.symbols("sigma1:4", nonnegative=True)}
    kinematic_variables.update(identity_mapping)
    return SympyDataTransformer.from_sympy(
        kinematic_variables, backend=backend, use_cse=False
    )


def create_phase_space_filter(
    decay: ThreeBodyDecay,
    x_mandelstam: FinalStateID = 1,
    y_mandelstam: FinalStateID = 2,
    outside_value=sp.nan,
) -> PositionalArgumentFunction:
    m0, m1, m2, m3 = create_mass_symbol_mapping(decay).values()
    sigma_x = sp.Symbol(f"sigma{x_mandelstam}", nonnegative=True)
    sigma_y = sp.Symbol(f"sigma{y_mandelstam}", nonnegative=True)
    in_phsp_expr = is_within_phasespace(sigma_x, sigma_y, m0, m1, m2, m3, outside_value)
    return create_function(in_phsp_expr.doit(), backend="jax", use_cse=True)


def generate_meshgrid_sample(
    decay: ThreeBodyDecay,
    resolution: int,
    x_mandelstam: FinalStateID = 1,
    y_mandelstam: FinalStateID = 2,
) -> DataSample:
    """Generate a `numpy.meshgrid` sample for plotting with `matplotlib.pyplot`."""
    boundaries = compute_dalitz_boundaries(decay)
    return generate_sub_meshgrid_sample(
        decay,
        resolution,
        x_range=boundaries[x_mandelstam - 1],
        y_range=boundaries[y_mandelstam - 1],
        x_mandelstam=x_mandelstam,
        y_mandelstam=y_mandelstam,
    )


def generate_sub_meshgrid_sample(  # noqa: PLR0917
    decay: ThreeBodyDecay,
    resolution: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    x_mandelstam: FinalStateID = 1,
    y_mandelstam: FinalStateID = 2,
) -> DataSample:
    sigma_x, sigma_y = jnp.meshgrid(
        jnp.linspace(*x_range, num=resolution),
        jnp.linspace(*y_range, num=resolution),
    )
    phsp: dict[str, Any] = {
        f"sigma{x_mandelstam}": sigma_x,
        f"sigma{y_mandelstam}": sigma_y,
    }
    z_mandelstam = __get_third_mandelstam_index(x_mandelstam, y_mandelstam)
    compute_sigma_z = __create_compute_compute_sigma_z(
        decay, x_mandelstam, y_mandelstam
    )
    phsp[f"sigma{z_mandelstam}"] = compute_sigma_z(phsp)
    return phsp


def generate_phasespace_sample(
    decay: ThreeBodyDecay, n_events: int, seed: int | None = None
) -> DataSample:
    r"""Generate a uniform distribution over Dalitz variables :math:`\sigma_{1,2,3}`."""
    boundaries = compute_dalitz_boundaries(decay)
    domain_generator = NumpyDomainGenerator(
        boundaries={
            "sigma1": boundaries[0],
            "sigma2": boundaries[1],
        }
    )
    phsp_filter = create_phase_space_filter(decay, outside_value=0)
    phsp_generator = IntensityDistributionGenerator(domain_generator, phsp_filter)
    rng = NumpyUniformRNG(seed)
    phsp = phsp_generator.generate(n_events, rng)
    compute_sigma_z = __create_compute_compute_sigma_z(decay)
    phsp["sigma3"] = compute_sigma_z(phsp)
    return phsp


def compute_dalitz_boundaries(
    decay: ThreeBodyDecay,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    m0, m1, m2, m3 = create_mass_symbol_mapping(decay).values()
    return (
        ((m2 + m3) ** 2, (m0 - m1) ** 2),
        ((m3 + m1) ** 2, (m0 - m2) ** 2),
        ((m1 + m2) ** 2, (m0 - m3) ** 2),
    )


def __create_compute_compute_sigma_z(
    decay: ThreeBodyDecay,
    x_mandelstam: FinalStateID = 1,
    y_mandelstam: FinalStateID = 2,
) -> PositionalArgumentFunction:
    masses = create_mass_symbol_mapping(decay)
    sigma_k = formulate_third_mandelstam(decay, x_mandelstam, y_mandelstam)
    return create_function(sigma_k.xreplace(masses), backend="jax", use_cse=True)


def __get_third_mandelstam_index(
    x_mandelstam: FinalStateID, y_mandelstam: FinalStateID
):
    if x_mandelstam == y_mandelstam:
        msg = "x_mandelstam and y_mandelstam must be different"
        raise ValueError(msg)
    return next(iter({1, 2, 3} - {x_mandelstam, y_mandelstam}))

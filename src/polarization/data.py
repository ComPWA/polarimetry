from __future__ import annotations

import sys

import numpy as np
import sympy as sp
from ampform.kinematics.phasespace import compute_third_mandelstam, is_within_phasespace
from tensorwaves.data import (
    IntensityDistributionGenerator,
    NumpyDomainGenerator,
    NumpyUniformRNG,
    SympyDataTransformer,
)
from tensorwaves.function import PositionalArgumentFunction
from tensorwaves.function.sympy import create_function
from tensorwaves.interface import DataSample

from polarization.amplitude import AmplitudeModel
from polarization.decay import ThreeBodyDecay

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal


def create_data_transformer(model: AmplitudeModel) -> SympyDataTransformer:
    kinematic_variables = {
        symbol: expression.doit().subs(model.parameter_defaults)
        for symbol, expression in model.variables.items()
    }
    identity_mapping = {s: s for s in sp.symbols("sigma1:4", nonnegative=True)}
    kinematic_variables.update(identity_mapping)
    return SympyDataTransformer.from_sympy(
        kinematic_variables, backend="jax", use_cse=False
    )


def create_phase_space_filter(
    decay: ThreeBodyDecay,
    x_mandelstam: Literal[1, 2, 3] = 1,
    y_mandelstam: Literal[1, 2, 3] = 2,
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
    x_mandelstam: Literal[1, 2, 3] = 1,
    y_mandelstam: Literal[1, 2, 3] = 2,
) -> DataSample:
    """Generate a `numpy.meshgrid` sample for plotting with `matplotlib.pyplot`."""
    boundaries = __compute_dalitz_boundaries(decay)
    sigma_x, sigma_y = np.meshgrid(
        np.linspace(*boundaries[x_mandelstam - 1], num=resolution),
        np.linspace(*boundaries[y_mandelstam - 1], num=resolution),
    )
    phsp = {
        f"sigma{x_mandelstam}": sigma_x,
        f"sigma{y_mandelstam}": sigma_y,
    }
    z_mandelstam = __get_third_mandelstam_index(x_mandelstam, y_mandelstam)
    compute_sigma_z = __create_compute_compute_sigma_z(
        decay, x_mandelstam, y_mandelstam
    )
    phsp[f"sigma{z_mandelstam}"] = compute_sigma_z(phsp)
    return phsp


def generate_phasespace_sample(decay: ThreeBodyDecay, n_events: int, seed=None):
    boundaries = __compute_dalitz_boundaries(decay)
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


def __compute_dalitz_boundaries(
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
    x_mandelstam: Literal[1, 2, 3] = 1,
    y_mandelstam: Literal[1, 2, 3] = 2,
) -> PositionalArgumentFunction:
    m0, m1, m2, m3 = create_mass_symbol_mapping(decay).values()
    sigma_x = sp.Symbol(f"sigma{x_mandelstam}", nonnegative=True)
    sigma_y = sp.Symbol(f"sigma{y_mandelstam}", nonnegative=True)
    sigma_k = compute_third_mandelstam(sigma_x, sigma_y, m0, m1, m2, m3)
    return create_function(sigma_k, backend="jax", use_cse=True)


def create_mass_symbol_mapping(decay: ThreeBodyDecay) -> dict[sp.Symbol, float]:
    return {
        sp.Symbol(f"m{i}"): decay.states[i].mass
        for i in sorted(decay.states)  # ensure that dict keys are sorted by state ID
    }


def __get_third_mandelstam_index(
    x_mandelstam: Literal[1, 2, 3], y_mandelstam: Literal[1, 2, 3]
):
    if x_mandelstam == y_mandelstam:
        raise ValueError(f"x_mandelstam and y_mandelstam must be different")
    return next(iter({1, 2, 3} - {x_mandelstam, y_mandelstam}))

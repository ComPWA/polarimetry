from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import sympy as sp
from ampform_dpd.io import cached

from polarimetry import lhcb
from polarimetry.data import create_data_transformer, generate_phasespace_sample
from polarimetry.function import (
    integrate_intensity,
    interference_intensity,
    sub_intensity,
)
from polarimetry.lhcb import load_model
from polarimetry.lhcb.particle import load_particles

if TYPE_CHECKING:
    from ampform_dpd import AmplitudeModel
    from tensorwaves.interface import DataSample, ParametrizedFunction


@pytest.fixture(scope="session")
def model() -> AmplitudeModel:
    data_dir = Path(lhcb.__file__).parent
    particles = load_particles(data_dir / "particle-definitions.yaml")
    return load_model(data_dir / "model-definitions.yaml", particles, model_id=0)


@pytest.fixture(scope="session")
def intensity_func(model: AmplitudeModel) -> ParametrizedFunction:
    unfolded_intensity_expr = cached.unfold(model)
    free_parameters = {
        symbol: value
        for symbol, value in model.parameter_defaults.items()
        if isinstance(symbol, sp.Indexed)
        if "production" in str(symbol)
    }
    fixed_parameters = {
        symbol: value
        for symbol, value in model.parameter_defaults.items()
        if symbol not in free_parameters
    }
    subs_intensity_expr = cached.xreplace(unfolded_intensity_expr, fixed_parameters)
    return cached.lambdify(
        subs_intensity_expr,
        parameters=free_parameters,  # type:ignore[arg-type]
        backend="jax",
    )


@pytest.fixture(scope="session")
def phsp(model: AmplitudeModel) -> DataSample:
    transformer = create_data_transformer(model)
    phsp = generate_phasespace_sample(model.decay, n_events=100_000, seed=0)
    return transformer(phsp)


@pytest.mark.slow
def test_interference_intensity(intensity_func: ParametrizedFunction, phsp: DataSample):
    K = "K(1430)"
    L = R"\Lambda(1405)"

    I_KL = interference_intensity(intensity_func, phsp, [K], [L])
    I_LK = interference_intensity(intensity_func, phsp, [L], [K])
    I_KK = interference_intensity(intensity_func, phsp, [K], [K])
    I_LL = interference_intensity(intensity_func, phsp, [L], [L])
    I_tot = integrate_intensity(intensity_func(phsp))

    assert pytest.approx(I_LK) == I_KL
    decay_rates_array = [
        I_KK / I_tot,
        I_KL / I_tot,
        I_LL / I_tot,
    ]
    decay_rates = [float(v) for v in decay_rates_array]
    expected = [
        -0.14700511274990016,
        +0.04779050812786789,
        -0.07778927301796905,
    ]
    np.testing.assert_allclose(
        decay_rates,
        expected,
        atol=1e-16,
        err_msg=str(decay_rates),
        rtol=1e-16,
    )


@pytest.mark.slow
def test_sub_intensity_all(intensity_func: ParametrizedFunction, phsp: DataSample):
    I_tot = integrate_intensity(intensity_func(phsp))
    np.testing.assert_allclose(
        I_tot,
        sub_intensity(intensity_func, phsp, ["K", "L", "D"]),
    )


@pytest.mark.slow
def test_total_intensity(intensity_func: ParametrizedFunction, phsp: DataSample):
    I_K = sub_intensity(intensity_func, phsp, non_zero_couplings=["K"])
    I_Λ = sub_intensity(intensity_func, phsp, non_zero_couplings=["L"])
    I_Δ = sub_intensity(intensity_func, phsp, non_zero_couplings=["D"])
    I_ΛΔ = interference_intensity(intensity_func, phsp, ["L"], ["D"])
    I_KΔ = interference_intensity(intensity_func, phsp, ["K"], ["D"])
    I_KΛ = interference_intensity(intensity_func, phsp, ["K"], ["L"])
    I_tot = integrate_intensity(intensity_func(phsp))
    np.testing.assert_allclose(I_tot, I_K + I_Λ + I_Δ + I_ΛΔ + I_KΔ + I_KΛ)

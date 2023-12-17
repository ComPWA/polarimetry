from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import sympy as sp
from ampform.sympy import PoolSum
from sympy.physics.matrices import msigma

from polarimetry.lhcb import ModelDefinition, ModelName, load_model
from polarimetry.lhcb.particle import (
    ResonanceJSON,
    _load_particles_json,  # pyright: ignore[reportPrivateUsage]
    load_particles,
)
from polarimetry.spin import create_spin_range

if TYPE_CHECKING:
    from polarimetry.amplitude import AmplitudeModel, DalitzPlotDecompositionBuilder


def formulate_polarimetry(
    builder: DalitzPlotDecompositionBuilder, reference_subsystem: Literal[1, 2, 3] = 1
) -> tuple[PoolSum, PoolSum, PoolSum]:
    half = sp.Rational(1, 2)
    if builder.decay.initial_state.spin != half:
        msg = (
            "Can only formulate polarimetry for an initial state with spin 1/2, but"
            f" got {builder.decay.initial_state.spin}"
        )
        raise ValueError(msg)
    model = builder.formulate(reference_subsystem)
    λ0, λ0_prime = sp.symbols(R"lambda \lambda^{\prime}", rational=True)
    λ = {
        sp.Symbol(f"lambda{i}", rational=True): create_spin_range(state.spin)
        for i, state in builder.decay.final_state.items()
    }
    ref = reference_subsystem
    return tuple(
        PoolSum(
            builder.formulate_aligned_amplitude(λ0, *λ, ref)[0].conjugate()
            * pauli_matrix[_to_index(λ0), _to_index(λ0_prime)]
            * builder.formulate_aligned_amplitude(λ0_prime, *λ, ref)[0],
            (λ0, [-half, +half]),
            (λ0_prime, [-half, +half]),
            *λ.items(),
        ).cleanup()
        / model.intensity
        for pauli_matrix in map(msigma, [1, 2, 3])
    )


def _to_index(helicity):
    """Symbolic conversion of half-value helicities to Pauli matrix indices."""
    return sp.Piecewise(
        (1, sp.LessThan(helicity, 0)),
        (0, True),
    )


def published_model(
    model_id: int | ModelName = 0,
    model_file: Path | str | None = None,
    particle_file: Path | str | None = None,
) -> AmplitudeModel:
    """Import model data and parameters, perform coupling conversions and return model."""
    src_dir = Path(__file__).parent
    if model_file is None:
        model_file = src_dir / "lhcb/model-definitions.yaml"
    if particle_file is None:
        particle_file = src_dir / "lhcb/particle-definitions.yaml"
    particles = load_particles(particle_file)
    return load_model(model_file, particles, model_id)


def expose_model_description() -> tuple[
    dict[ModelName, ModelDefinition],
    dict[str, ResonanceJSON],
]:
    """Load all published model and particle definitions.

    Returns a `tuple` of:

    1. all 18 model definitions from :download:`model-definitions.yaml
       <../../data/model-definitions.yaml>`,
    2. particle definitions from :download:`particle-definitions.yaml
       <../../data/particle-definitions.yaml>`.
    """
    src_dir = Path(__file__).parent
    particles = _load_particles_json(src_dir / "lhcb/particle-definitions.yaml")
    model = load_model(src_dir / "lhcb/model-definitions.yaml", particles, model_id=0)
    return model, particles

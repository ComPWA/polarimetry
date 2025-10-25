from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from ampform_dpd import DynamicsBuilder, create_mass_symbol_mapping
from ampform_dpd.dynamics.builder import get_mandelstam_s
from ampform_dpd.io import cached

from polarimetry.lhcb import (
    ModelName,
    ResonanceName,
    load_model_parameters,
    load_three_body_decay,
)
from polarimetry.lhcb.dynamics import (
    formulate_breit_wigner,
    formulate_bugg_breit_wigner,
    formulate_exponential_bugg_breit_wigner,
    formulate_flatte_1405,
)
from polarimetry.lhcb.particle import load_particles

if TYPE_CHECKING:
    import sympy as sp
    from ampform_dpd.decay import ThreeBodyDecay

LS_MODEL_ID: ModelName = "Alternative amplitude model obtained using LS couplings"
EXP_MODEL_ID: ModelName = "Alternative amplitude model with an additional overall exponential form factor exp(-alpha q^2) multiplying Bugg lineshapes. The exponential parameter is indicated as alpha"
REPO_DIR = Path(__file__).parent.parent


@pytest.mark.parametrize(
    ("resonance", "dynamics_builder", "expected"),
    [
        (
            "D(1232)",
            formulate_breit_wigner,
            [
                [3.187925 + 0.672379j, -1.446337 + 1.42195j, -0.697237 + 0.554528j],
                [3.22941 + 0.681129j, -1.359803 + 1.336876j, -0.363739 + 0.28929j],
            ],
        ),
        (
            "D(1600)",
            formulate_breit_wigner,
            [
                [0.271035 + 0.004131j, 1.174441 + 0.499726j, -0.623537 + 1.317549j],
                [0.368883 + 0.005622j, 1.483497 + 0.631229j, -0.437039 + 0.923475j],
            ],
        ),
        (
            "D(1700)",
            formulate_breit_wigner,
            [
                [
                    0.054407 + 4.219208e-05j,
                    0.70962 + 1.189754e-01j,
                    -0.190334 + 1.316309j,
                ],
                [
                    0.087835 + 6.811540e-05j,
                    1.063241 + 1.782638e-01j,
                    -0.158244 + 1.094377j,
                ],
            ],
        ),
        (
            "K(1430)",
            formulate_bugg_breit_wigner,
            [
                [0.716759 + 0.021065j, 1.082562 + 0.137335j, 2.306636 + 1.878309j],
                [0.716759 + 0.021065j, 1.082562 + 0.137335j, 2.306636 + 1.878309j],
            ],
        ),
        (
            "K(1430)",
            formulate_exponential_bugg_breit_wigner,
            [[0.586534 + 0.018307j, 0.955479 + 0.136514j, 1.885291 + 1.870264j]],
        ),
        (
            "K(700)",
            formulate_bugg_breit_wigner,
            [
                [3.224889 + 2.699736j, -1.734052 + 1.650639j, -0.950293 + 0.310209j],
                [3.224889 + 2.699736j, -1.734052 + 1.650639j, -0.950293 + 0.310209j],
            ],
        ),
        (
            "K(700)",
            formulate_exponential_bugg_breit_wigner,
            [[4.214162 + 3.572103j, -2.001794 + 1.952961j, -0.976101 + 0.33192j]],
        ),
        (
            "K(892)",
            formulate_breit_wigner,
            [
                [1.610608 + 0.031587j, -5.798347 + 2.097499j, -1.838173 + 0.294361j],
                [1.627073 + 0.03191j, -5.744577 + 2.078048j, -1.559624 + 0.249755j],
                [1.627073 + 0.03191j, -5.744577 + 2.078048j, -1.559624 + 0.249755j],
                [1.671756 + 0.032786j, -5.597768 + 2.024941j, -0.960289 + 0.153779j],
            ],
        ),
        (
            "L(1405)",
            formulate_flatte_1405,
            [
                [-2.304177 + 1.657209j, -0.746962 + 0.224641j, -0.413249 + 0.079752j],
                [-2.286201 + 1.64428j, -0.702741 + 0.211342j, -0.258851 + 0.049955j],
            ],
        ),
        (
            "L(1520)",
            formulate_breit_wigner,
            [
                [16.355223 + 5.327755j, -2.539074 + 2.228754j, -0.726684 + 1.09592j],
                [16.422337 + 5.349617j, -2.177679 + 1.911528j, -0.232965 + 0.351336j],
            ],
        ),
        (
            "L(1600)",
            formulate_breit_wigner,
            [
                [1.500917 + 0.395826j, -0.716189 + 1.065852j, -0.52262 + 0.453102j],
                [1.524289 + 0.40199j, -0.689663 + 1.026375j, -0.33507 + 0.2905j],
            ],
        ),
        (
            "L(1670)",
            formulate_breit_wigner,
            [
                [1.886101 + 0.104917j, -2.372729 + 0.336997j, -0.657148 + 0.03091j],
                [1.928693 + 0.107286j, -2.300618 + 0.326756j, -0.424229 + 0.019954j],
            ],
        ),
        (
            "L(1690)",
            formulate_breit_wigner,
            [
                [0.465468 + 0.003831j, -2.504197 + 1.88068j, -0.749837 + 0.515093j],
                [0.502314 + 0.004135j, -2.308309 + 1.733567j, -0.258356 + 0.177475j],
            ],
        ),
        (
            "L(2000)",
            formulate_breit_wigner,
            [
                [0.581031 + 0.052384j, 1.125157 + 0.45146j, -1.331273 + 1.465054j],
                [0.71857 + 0.064784j, 1.319412 + 0.529404j, -1.039383 + 1.143831j],
            ],
        ),
    ],
)
def test_dynamics_builder(  # noqa: PLR0914
    dynamics_builder: DynamicsBuilder,
    resonance: ResonanceName,
    expected: list[list[complex]],
) -> None:
    min_ls = dynamics_builder is formulate_exponential_bugg_breit_wigner
    model_id = EXP_MODEL_ID if min_ls else LS_MODEL_ID
    particles = load_particles(REPO_DIR / "data/particle-definitions.yaml")
    decay = load_three_body_decay([resonance], particles, min_ls)
    parameter_defaults = load_parameters(decay, model_id)
    parameter_defaults.update(create_mass_symbol_mapping(decay))
    arrays = []
    for chain in sorted(decay.chains):
        expr, parameters = dynamics_builder(chain)
        func = cached.lambdify(
            expr.doit(),
            parameters=parameters | parameter_defaults,
            backend="jax",
        )
        child1, child2 = chain.decay_products
        x_min = child1.mass + child2.mass
        x_max = chain.parent.mass - chain.spectator.mass
        x_diff = x_max - x_min
        x = np.linspace(x_min + 0.1 * x_diff, x_max - 0.1 * x_diff, num=3)
        sigma = get_mandelstam_s(chain.decay_node)
        z = func({sigma.name: x**2})
        arrays.append(z.tolist())
    np.testing.assert_allclose(arrays, expected, atol=1e-6, rtol=1e-6)


def load_parameters(
    decay: ThreeBodyDecay, model_id: ModelName
) -> dict[sp.Symbol, complex]:
    particles = load_particles(REPO_DIR / "data/particle-definitions.yaml")
    return load_model_parameters(
        decay=decay,
        filename=REPO_DIR / "data/model-definitions.yaml",
        model_id=model_id,
        particle_definitions=particles,
    )

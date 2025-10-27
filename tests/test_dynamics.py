from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from ampform_dpd import DynamicsBuilder, create_mass_symbol_mapping
from ampform_dpd.dynamics.builder import get_mandelstam_s
from tensorwaves.function.sympy import create_parametrized_function

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
                [
                    (3.1879237244582677 + 0.6723793982835011j),
                    (-1.4463367213423046 + 1.4219501754895838j),
                    (-0.6972370370457636 + 0.5545283926872141j),
                ],
                [
                    (3.229409123039008 + 0.6811292711619713j),
                    (-1.3598033106780638 + 1.3368757964296827j),
                    (-0.3637396197523502 + 0.2892903503126588j),
                ],
            ],
        ),
        (
            "D(1600)",
            formulate_breit_wigner,
            [
                [
                    (0.2710347852652632 + 0.00413056821040814j),
                    (1.1744416876390549 + 0.4997258940032302j),
                    (-0.6235368677063432 + 1.3175500554559383j),
                ],
                [
                    (0.36888301140431373 + 0.005621774484684802j),
                    (1.4834968059344198 + 0.6312290983870059j),
                    (-0.43703958234775686 + 0.9234763103533312j),
                ],
            ],
        ),
        (
            "D(1700)",
            formulate_breit_wigner,
            [
                [
                    (0.05440680189842112 + 4.219215707636137e-05j),
                    (0.7096200532917741 + 0.11897536374158689j),
                    (-0.19033478821404695 + 1.3163096550591395j),
                ],
                [
                    (0.08783500566719082 + 6.81155338413808e-05j),
                    (1.0632413313758629 + 0.17826373924852945j),
                    (-0.1582441348439172 + 1.0943784082039711j),
                ],
            ],
        ),
        (
            "K(700)",
            formulate_bugg_breit_wigner,
            [
                [
                    (3.224888626193087 + 2.699735741237637j),
                    (-1.734051582511575 + 1.650638897678022j),
                    (-0.9502933990518189 + 0.3102090462388432j),
                ],
                [
                    (3.224888626193087 + 2.699735741237637j),
                    (-1.734051582511575 + 1.650638897678022j),
                    (-0.9502933990518189 + 0.3102090462388432j),
                ],
            ],
        ),
        (
            "K(700)",
            formulate_exponential_bugg_breit_wigner,
            [
                [
                    (4.214161746712885 + 3.5721028984027168j),
                    (-2.001794401493292 + 1.95296089163316j),
                    (-0.976100865026633 + 0.33192019359205693j),
                ]
            ],
        ),
        (
            "K(892)",
            formulate_breit_wigner,
            [
                [
                    (1.6106080680274741 + 0.03158685433213549j),
                    (-5.798347035134785 + 2.097498626516587j),
                    (-1.8381729460342473 + 0.29436136284355285j),
                ],
                [
                    (1.627073261656379 + 0.03190976571140331j),
                    (-5.744577658470813 + 2.0780480498232734j),
                    (-1.5596243472853126 + 0.24975514375911056j),
                ],
                [
                    (1.627073261656379 + 0.03190976571140331j),
                    (-5.744577658470813 + 2.0780480498232734j),
                    (-1.5596243472853126 + 0.24975514375911056j),
                ],
                [
                    (1.6717559227313357 + 0.03278607121027949j),
                    (-5.5977681625977675 + 2.024941067076743j),
                    (-0.9602905295863794 + 0.15377901716193432j),
                ],
            ],
        ),
        (
            "K(1430)",
            formulate_bugg_breit_wigner,
            [
                [
                    (0.7167591636230847 + 0.021065036045185274j),
                    (1.0825620220616838 + 0.13733511891708203j),
                    (2.306635778206944 + 1.8783081927150984j),
                ],
                [
                    (0.7167591636230847 + 0.021065036045185274j),
                    (1.0825620220616838 + 0.13733511891708203j),
                    (2.306635778206944 + 1.8783081927150984j),
                ],
            ],
        ),
        (
            "K(1430)",
            formulate_exponential_bugg_breit_wigner,
            [
                [
                    (0.5865336530761671 + 0.018306556196762j),
                    (0.9554786256660777 + 0.1365144676524909j),
                    (1.88529058820957 + 1.8702634495849284j),
                ]
            ],
        ),
        (
            "L(1405)",
            formulate_flatte_1405,
            [
                [
                    (-2.3041770091552602 + 1.6572083133216777j),
                    (-0.7469620689351583 + 0.2246413308846193j),
                    (-0.41324873584663085 + 0.07975156887710629j),
                ],
                [
                    (-2.286201762761729 + 1.644280171239275j),
                    (-0.702741374356551 + 0.211342401667314j),
                    (-0.2588509546461646 + 0.049954828527363826j),
                ],
            ],
        ),
        (
            "L(1520)",
            formulate_breit_wigner,
            [
                [
                    (16.35520530261141 + 5.327731299952113j),
                    (-2.5390775187062373 + 2.2287535598743053j),
                    (-0.7266872291636765 + 1.0959216306941388j),
                ],
                [
                    (16.42231759922914 + 5.349593225662386j),
                    (-2.1776814474732817 + 1.9115270182067412j),
                    (-0.23296608483836545 + 0.35133763377997224j),
                ],
            ],
        ),
        (
            "L(1600)",
            formulate_breit_wigner,
            [
                [
                    (1.500916330518528 + 0.3958255988816235j),
                    (-0.7161889637173755 + 1.0658517878040505j),
                    (-0.5226198956883473 + 0.4531021629059485j),
                ],
                [
                    (1.5242890504510038 + 0.40198951400244076j),
                    (-0.6896626872456154 + 1.0263746656567632j),
                    (-0.3350705873438278 + 0.2905002452149088j),
                ],
            ],
        ),
        (
            "L(1670)",
            formulate_breit_wigner,
            [
                [
                    (1.8861010191054925 + 0.1049170973486188j),
                    (-2.3727294353163586 + 0.336997489157645j),
                    (-0.657148215606705 + 0.03090971404562307j),
                ],
                [
                    (1.9286929136646251 + 0.10728633309074354j),
                    (-2.3006185210233765 + 0.3267556146750801j),
                    (-0.4242296210381687 + 0.019954122927757733j),
                ],
            ],
        ),
        (
            "L(1690)",
            formulate_breit_wigner,
            [
                [
                    (0.4654680460487038 + 0.003831435962280073j),
                    (-2.504196242196581 + 1.8806796658542846j),
                    (-0.7498382369129406 + 0.515093279641933j),
                ],
                [
                    (0.5023142911137491 + 0.004134730741837138j),
                    (-2.308309109724075 + 1.7335662165822747j),
                    (-0.2583568441894517 + 0.17747544422295913j),
                ],
            ],
        ),
        (
            "L(2000)",
            formulate_breit_wigner,
            [
                [
                    (0.5810314284422579 + 0.052383661274103814j),
                    (1.1251566026793989 + 0.4514599573685283j),
                    (-1.3312729961106082 + 1.4650550204028105j),
                ],
                [
                    (0.7185693849654652 + 0.06478357868676422j),
                    (1.319411658565389 + 0.5294032223683239j),
                    (-1.0393832872156312 + 1.1438327882461754j),
                ],
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
    arrays: list[list[complex]] = []
    for chain in sorted(decay.chains):
        expr, parameters = dynamics_builder(chain)
        func = create_parametrized_function(
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
    np.testing.assert_allclose(
        arrays, expected, err_msg=str(arrays), rtol=1e-16, atol=1e-16
    )


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

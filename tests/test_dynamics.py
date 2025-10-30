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
                    (3.187923724458267 + 0.6723793982835005j),
                    (-1.4463367213423055 + 1.4219501754895831j),
                    (-0.6972370370457637 + 0.5545283926872138j),
                ],
                [
                    (3.2294091230390083 + 0.681129271161971j),
                    (-1.3598033106780647 + 1.3368757964296825j),
                    (-0.3637396197523503 + 0.2892903503126587j),
                ],
            ],
        ),
        (
            "D(1600)",
            formulate_breit_wigner,
            [
                [
                    (0.27103478526526326 + 0.004130568210408139j),
                    (1.1744416876390549 + 0.4997258940032301j),
                    (-0.6235368677063435 + 1.3175500554559385j),
                ],
                [
                    (0.36888301140431384 + 0.005621774484684806j),
                    (1.48349680593442 + 0.6312290983870059j),
                    (-0.437039582347757 + 0.9234763103533314j),
                ],
            ],
        ),
        (
            "D(1700)",
            formulate_breit_wigner,
            [
                [
                    (0.05440680189842114 + 4.2192157076361384e-05j),
                    (0.7096200532917742 + 0.11897536374158696j),
                    (-0.19033478821404676 + 1.3163096550591393j),
                ],
                [
                    (0.08783500566719082 + 6.811553384138082e-05j),
                    (1.0632413313758626 + 0.1782637392485295j),
                    (-0.15824413484391714 + 1.0943784082039707j),
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
                    (1.610608068027474 + 0.03158685433213548j),
                    (-5.798347035134786 + 2.097498626516587j),
                    (-1.8381729460342475 + 0.29436136284355296j),
                ],
                [
                    (1.627073261656379 + 0.031909765711403304j),
                    (-5.744577658470811 + 2.0780480498232725j),
                    (-1.5596243472853124 + 0.24975514375911054j),
                ],
                [
                    (1.627073261656379 + 0.031909765711403304j),
                    (-5.744577658470811 + 2.0780480498232725j),
                    (-1.5596243472853124 + 0.24975514375911054j),
                ],
                [
                    (1.6717559227313354 + 0.0327860712102795j),
                    (-5.597768162597763 + 2.0249410670767403j),
                    (-0.9602905295863792 + 0.1537790171619342j),
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
                    (-2.30417700915526 + 1.6572083133216777j),
                    (-0.7469620689351583 + 0.22464133088461932j),
                    (-0.41324873584663085 + 0.07975156887710633j),
                ],
                [
                    (-2.2862017627617295 + 1.6442801712392754j),
                    (-0.7027413743565509 + 0.21134240166731402j),
                    (-0.25885095464616464 + 0.04995482852736384j),
                ],
            ],
        ),
        (
            "L(1520)",
            formulate_breit_wigner,
            [
                [
                    (16.3552053026114 + 5.327731299952118j),
                    (-2.5390775187062338 + 2.228753559874305j),
                    (-0.7266872291636755 + 1.0959216306941384j),
                ],
                [
                    (16.42231759922913 + 5.349593225662392j),
                    (-2.17768144747328 + 1.9115270182067425j),
                    (-0.23296608483836517 + 0.3513376337799723j),
                ],
            ],
        ),
        (
            "L(1600)",
            formulate_breit_wigner,
            [
                [
                    (1.500916330518528 + 0.3958255988816238j),
                    (-0.7161889637173754 + 1.0658517878040505j),
                    (-0.5226198956883473 + 0.4531021629059487j),
                ],
                [
                    (1.5242890504510038 + 0.401989514002441j),
                    (-0.6896626872456146 + 1.0263746656567625j),
                    (-0.3350705873438277 + 0.29050024521490875j),
                ],
            ],
        ),
        (
            "L(1670)",
            formulate_breit_wigner,
            [
                [
                    (1.8861010191054925 + 0.10491709734861879j),
                    (-2.3727294353163586 + 0.33699748915764494j),
                    (-0.657148215606705 + 0.03090971404562308j),
                ],
                [
                    (1.9286929136646254 + 0.10728633309074352j),
                    (-2.3006185210233756 + 0.32675561467507996j),
                    (-0.42422962103816864 + 0.019954122927757733j),
                ],
            ],
        ),
        (
            "L(1690)",
            formulate_breit_wigner,
            [
                [
                    (0.46546804604870373 + 0.0038314359622800775j),
                    (-2.5041962421965787 + 1.8806796658542855j),
                    (-0.74983823691294 + 0.5150932796419333j),
                ],
                [
                    (0.502314291113749 + 0.0041347307418371435j),
                    (-2.3083091097240738 + 1.7335662165822758j),
                    (-0.25835684418945165 + 0.17747544422295927j),
                ],
            ],
        ),
        (
            "L(2000)",
            formulate_breit_wigner,
            [
                [
                    (0.5810314284422579 + 0.052383661274103814j),
                    (1.1251566026793989 + 0.45145995736852834j),
                    (-1.3312729961106082 + 1.4650550204028105j),
                ],
                [
                    (0.7185693849654651 + 0.06478357868676422j),
                    (1.3194116585653886 + 0.5294032223683237j),
                    (-1.039383287215631 + 1.1438327882461752j),
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
        dynamics = dynamics_builder(chain)
        func = create_parametrized_function(
            dynamics.expression.doit(),
            parameters=dynamics.parameters | parameter_defaults,
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

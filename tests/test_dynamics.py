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
                    (3.187923724458265 + 0.6723793982835j),
                    (-1.4463367213423053 + 1.4219501754895836j),
                    (-0.6972370370457632 + 0.5545283926872139j),
                ],
                [
                    (3.2294091230390074 + 0.681129271161971j),
                    (-1.3598033106780643 + 1.3368757964296825j),
                    (-0.3637396197523502 + 0.2892903503126587j),
                ],
            ],
        ),
        (
            "D(1600)",
            formulate_breit_wigner,
            [
                [
                    (0.2710347852652631 + 0.004130568210408139j),
                    (1.1744416876390544 + 0.49972589400323003j),
                    (-0.6235368677063433 + 1.3175500554559378j),
                ],
                [
                    (0.3688830114043136 + 0.005621774484684802j),
                    (1.4834968059344196 + 0.631229098387006j),
                    (-0.4370395823477569 + 0.9234763103533312j),
                ],
            ],
        ),
        (
            "D(1700)",
            formulate_breit_wigner,
            [
                [
                    (0.054406801898421134 + 4.2192157076361384e-05j),
                    (0.7096200532917745 + 0.11897536374158701j),
                    (-0.19033478821404676 + 1.316309655059139j),
                ],
                [
                    (0.08783500566719082 + 6.811553384138082e-05j),
                    (1.0632413313758629 + 0.17826373924852953j),
                    (-0.15824413484391706 + 1.094378408203971j),
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
                    (1.6106080680274737 + 0.03158685433213549j),
                    (-5.798347035134785 + 2.097498626516587j),
                    (-1.8381729460342475 + 0.2943613628435529j),
                ],
                [
                    (1.6270732616563792 + 0.03190976571140332j),
                    (-5.744577658470812 + 2.078048049823273j),
                    (-1.559624347285313 + 0.24975514375911062j),
                ],
                [
                    (1.6270732616563792 + 0.03190976571140332j),
                    (-5.744577658470812 + 2.078048049823273j),
                    (-1.559624347285313 + 0.24975514375911062j),
                ],
                [
                    (1.6717559227313352 + 0.032786071210279495j),
                    (-5.597768162597767 + 2.0249410670767416j),
                    (-0.9602905295863792 + 0.15377901716193426j),
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
                    (-2.286201762761729 + 1.644280171239275j),
                    (-0.7027413743565507 + 0.211342401667314j),
                    (-0.2588509546461646 + 0.04995482852736384j),
                ],
            ],
        ),
        (
            "L(1520)",
            formulate_breit_wigner,
            [
                [
                    (16.355205302611402 + 5.327731299952124j),
                    (-2.539077518706233 + 2.228753559874306j),
                    (-0.7266872291636742 + 1.095921630694138j),
                ],
                [
                    (16.422317599229135 + 5.349593225662396j),
                    (-2.177681447473277 + 1.9115270182067416j),
                    (-0.23296608483836473 + 0.35133763377997196j),
                ],
            ],
        ),
        (
            "L(1600)",
            formulate_breit_wigner,
            [
                [
                    (1.500916330518528 + 0.39582559888162366j),
                    (-0.716188963717375 + 1.0658517878040503j),
                    (-0.5226198956883471 + 0.45310216290594857j),
                ],
                [
                    (1.5242890504510036 + 0.4019895140024409j),
                    (-0.6896626872456146 + 1.0263746656567625j),
                    (-0.33507058734382783 + 0.2905002452149088j),
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
                    (1.9286929136646251 + 0.10728633309074352j),
                    (-2.300618521023376 + 0.32675561467507996j),
                    (-0.4242296210381687 + 0.01995412292775774j),
                ],
            ],
        ),
        (
            "L(1690)",
            formulate_breit_wigner,
            [
                [
                    (0.46546804604870395 + 0.003831435962280081j),
                    (-2.504196242196578 + 1.8806796658542866j),
                    (-0.7498382369129396 + 0.5150932796419334j),
                ],
                [
                    (0.5023142911137491 + 0.0041347307418371435j),
                    (-2.3083091097240733 + 1.7335662165822754j),
                    (-0.25835684418945143 + 0.17747544422295922j),
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

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
                    (3.187923724458266 + 0.6723793982834997j),
                    (-1.4463367213423062 + 1.4219501754895827j),
                    (-0.6972370370457638 + 0.5545283926872138j),
                ],
                [
                    (3.229409123039007 + 0.6811292711619702j),
                    (-1.3598033106780651 + 1.3368757964296818j),
                    (-0.36373961975235053 + 0.2892903503126588j),
                ],
            ],
        ),
        (
            "D(1600)",
            formulate_breit_wigner,
            [
                [
                    (0.27103478526526315 + 0.004130568210408137j),
                    (1.1744416876390549 + 0.4997258940032302j),
                    (-0.6235368677063434 + 1.3175500554559387j),
                ],
                [
                    (0.368883011404314 + 0.005621774484684806j),
                    (1.4834968059344207 + 0.6312290983870065j),
                    (-0.4370395823477572 + 0.9234763103533322j),
                ],
            ],
        ),
        (
            "D(1700)",
            formulate_breit_wigner,
            [
                [
                    (0.054406801898421044 + 4.219215707636125e-05j),
                    (0.7096200532917742 + 0.11897536374158703j),
                    (-0.19033478821404692 + 1.31630965505914j),
                ],
                [
                    (0.08783500566719081 + 6.81155338413807e-05j),
                    (1.0632413313758642 + 0.17826373924852987j),
                    (-0.1582441348439175 + 1.0943784082039731j),
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
                    (-0.9761008650266332 + 0.331920193592057j),
                ]
            ],
        ),
        (
            "K(892)",
            formulate_breit_wigner,
            [
                [
                    (1.6106080680274735 + 0.03158685433213547j),
                    (-5.798347035134785 + 2.097498626516587j),
                    (-1.8381729460342475 + 0.2943613628435529j),
                ],
                [
                    (1.6270732616563786 + 0.03190976571140328j),
                    (-5.744577658470813 + 2.078048049823272j),
                    (-1.5596243472853135 + 0.24975514375911062j),
                ],
                [
                    (1.6270732616563786 + 0.03190976571140328j),
                    (-5.744577658470813 + 2.078048049823272j),
                    (-1.5596243472853135 + 0.24975514375911062j),
                ],
                [
                    (1.671755922731335 + 0.032786071210279474j),
                    (-5.597768162597766 + 2.02494106707674j),
                    (-0.960290529586381 + 0.15377901716193448j),
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
                    (-2.3041770091552616 + 1.6572083133216768j),
                    (-0.7469620689351584 + 0.22464133088461913j),
                    (-0.41324873584663085 + 0.07975156887710626j),
                ],
                [
                    (-2.2862017627617295 + 1.6442801712392745j),
                    (-0.702741374356551 + 0.21134240166731394j),
                    (-0.2588509546461649 + 0.04995482852736389j),
                ],
            ],
        ),
        (
            "L(1520)",
            formulate_breit_wigner,
            [
                [
                    (16.355205302611402 + 5.327731299952119j),
                    (-2.5390775187062347 + 2.2287535598743036j),
                    (-0.7266872291636761 + 1.0959216306941393j),
                ],
                [
                    (16.422317599229142 + 5.349593225662395j),
                    (-2.17768144747328 + 1.9115270182067408j),
                    (-0.23296608483836573 + 0.35133763377997296j),
                ],
            ],
        ),
        (
            "L(1600)",
            formulate_breit_wigner,
            [
                [
                    (1.5009163305185282 + 0.3958255988816238j),
                    (-0.7161889637173751 + 1.0658517878040503j),
                    (-0.5226198956883471 + 0.4531021629059486j),
                ],
                [
                    (1.5242890504510036 + 0.4019895140024409j),
                    (-0.6896626872456147 + 1.0263746656567627j),
                    (-0.335070587343828 + 0.29050024521490914j),
                ],
            ],
        ),
        (
            "L(1670)",
            formulate_breit_wigner,
            [
                [
                    (1.8861010191054925 + 0.10491709734861884j),
                    (-2.3727294353163586 + 0.33699748915764505j),
                    (-0.657148215606705 + 0.030909714045623083j),
                ],
                [
                    (1.9286929136646251 + 0.10728633309074348j),
                    (-2.300618521023376 + 0.3267556146750799j),
                    (-0.4242296210381692 + 0.019954122927757757j),
                ],
            ],
        ),
        (
            "L(1690)",
            formulate_breit_wigner,
            [
                [
                    (0.4654680460487043 + 0.0038314359622800822j),
                    (-2.504196242196581 + 1.8806796658542861j),
                    (-0.749838236912941 + 0.515093279641934j),
                ],
                [
                    (0.5023142911137499 + 0.0041347307418371495j),
                    (-2.3083091097240773 + 1.7335662165822763j),
                    (-0.25835684418945254 + 0.17747544422295988j),
                ],
            ],
        ),
        (
            "L(2000)",
            formulate_breit_wigner,
            [
                [
                    (0.5810314284422579 + 0.05238366127410383j),
                    (1.1251566026793989 + 0.45145995736852834j),
                    (-1.3312729961106085 + 1.4650550204028105j),
                ],
                [
                    (0.718569384965465 + 0.06478357868676421j),
                    (1.3194116585653883 + 0.5294032223683235j),
                    (-1.0393832872156321 + 1.1438327882461765j),
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
    np.testing.assert_array_equal(arrays, expected, err_msg=str(arrays))


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

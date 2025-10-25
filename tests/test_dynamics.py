from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from ampform_dpd import DynamicsBuilder, create_mass_symbol_mapping
from ampform_dpd.dynamics.builder import get_mandelstam_s
from tensorwaves.function.sympy import create_function

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
                    (3.1879237244582153 + 0.6723793982834547j),
                    (-1.4463367213423004 + 1.4219501754895698j),
                    (-0.6972370370457387 + 0.5545283926871928j),
                ],
                [
                    (3.2294091230389665 + 0.6811292711619269j),
                    (-1.3598033106780696 + 1.3368757964296794j),
                    (-0.36373961975234714 + 0.28929035031265554j),
                ],
            ],
        ),
        (
            "D(1600)",
            formulate_breit_wigner,
            [
                [
                    (0.2710347852652575 + 0.004130568210407858j),
                    (1.1744416876390438 + 0.49972589400322515j),
                    (-0.6235368677063182 + 1.3175500554558859j),
                ],
                [
                    (0.3688830114043073 + 0.005621774484684442j),
                    (1.4834968059344165 + 0.6312290983870044j),
                    (-0.437039582347751 + 0.9234763103533191j),
                ],
            ],
        ),
        (
            "D(1700)",
            formulate_breit_wigner,
            [
                [
                    (0.05440680189841888 + 4.219215707635592e-05j),
                    (0.7096200532917663 + 0.11897536374158432j),
                    (-0.19033478821404032 + 1.316309655059095j),
                ],
                [
                    (0.08783500566718742 + 6.811553384137221e-05j),
                    (1.0632413313758573 + 0.17826373924852665j),
                    (-0.15824413484391583 + 1.0943784082039625j),
                ],
            ],
        ),
        (
            "K(700)",
            formulate_bugg_breit_wigner,
            [
                [
                    (3.224888626193087 + 2.6997357412376393j),
                    (-1.7340515825115743 + 1.6506388976780213j),
                    (-0.9502933990518186 + 0.31020904623884327j),
                ],
                [
                    (3.224888626193087 + 2.6997357412376393j),
                    (-1.7340515825115743 + 1.6506388976780213j),
                    (-0.9502933990518186 + 0.31020904623884327j),
                ],
            ],
        ),
        (
            "K(700)",
            formulate_exponential_bugg_breit_wigner,
            [
                [
                    (4.214161746712877 + 3.5721028984027114j),
                    (-2.001794401493287 + 1.9529608916331551j),
                    (-0.9761008650266311 + 0.33192019359205643j),
                ]
            ],
        ),
        (
            "K(892)",
            formulate_breit_wigner,
            [
                [
                    (1.6106080680274744 + 0.03158685433213535j),
                    (-5.798347035134794 + 2.097498626516585j),
                    (-1.8381729460342502 + 0.29436136284355274j),
                ],
                [
                    (1.6270732616563734 + 0.031909765711403054j),
                    (-5.744577658470786 + 2.0780480498232587j),
                    (-1.5596243472852696 + 0.2497551437591032j),
                ],
                [
                    (1.6270732616563734 + 0.031909765711403054j),
                    (-5.744577658470786 + 2.0780480498232587j),
                    (-1.5596243472852696 + 0.2497551437591032j),
                ],
                [
                    (1.6717559227313314 + 0.032786071210279266j),
                    (-5.597768162597745 + 2.0249410670767287j),
                    (-0.9602905295863287 + 0.15377901716192585j),
                ],
            ],
        ),
        (
            "K(1430)",
            formulate_bugg_breit_wigner,
            [
                [
                    (0.7167591636230847 + 0.021065036045185313j),
                    (1.0825620220616838 + 0.13733511891708228j),
                    (2.3066357782069407 + 1.878308192715099j),
                ],
                [
                    (0.7167591636230847 + 0.021065036045185313j),
                    (1.0825620220616838 + 0.13733511891708228j),
                    (2.3066357782069407 + 1.878308192715099j),
                ],
            ],
        ),
        (
            "K(1430)",
            formulate_exponential_bugg_breit_wigner,
            [
                [
                    (0.5865336530761669 + 0.018306556196762025j),
                    (0.9554786256660774 + 0.13651446765249114j),
                    (1.8852905882095659 + 1.8702634495849277j),
                ]
            ],
        ),
        (
            "L(1405)",
            formulate_flatte_1405,
            [
                [
                    (-2.3041770091552536 + 1.6572083133216797j),
                    (-0.7469620689351584 + 0.22464133088461913j),
                    (-0.4132487358466309 + 0.0797515688771062j),
                ],
                [
                    (-2.286201762761687 + 1.6442801712392512j),
                    (-0.702741374356524 + 0.21134240166730564j),
                    (-0.2588509546460942 + 0.04995482852735018j),
                ],
            ],
        ),
        (
            "L(1520)",
            formulate_breit_wigner,
            [
                [
                    (16.35520530261135 + 5.327731299952243j),
                    (-2.539077518706156 + 2.228753559874233j),
                    (-0.7266872291634848 + 1.095921630693847j),
                ],
                [
                    (16.422317599229224 + 5.349593225662564j),
                    (-2.177681447473244 + 1.9115270182067083j),
                    (-0.23296608483827952 + 0.35133763377984184j),
                ],
            ],
        ),
        (
            "L(1600)",
            formulate_breit_wigner,
            [
                [
                    (1.500916330518535 + 0.39582559888163293j),
                    (-0.7161889637173738 + 1.0658517878040497j),
                    (-0.5226198956883471 + 0.45310216290594846j),
                ],
                [
                    (1.5242890504509903 + 0.40198951400244487j),
                    (-0.6896626872455885 + 1.0263746656567245j),
                    (-0.3350705873437372 + 0.2905002452148303j),
                ],
            ],
        ),
        (
            "L(1670)",
            formulate_breit_wigner,
            [
                [
                    (1.8861010191054925 + 0.10491709734861931j),
                    (-2.3727294353163586 + 0.3369974891576448j),
                    (-0.657148215606705 + 0.03090971404562305j),
                ],
                [
                    (1.928692913664607 + 0.10728633309074305j),
                    (-2.3006185210233014 + 0.3267556146750693j),
                    (-0.4242296210380559 + 0.019954122927752414j),
                ],
            ],
        ),
        (
            "L(1690)",
            formulate_breit_wigner,
            [
                [
                    (0.4654680460487037 + 0.0038314359622801963j),
                    (-2.5041962421964885 + 1.8806796658542273j),
                    (-0.7498382369127375 + 0.5150932796417952j),
                ],
                [
                    (0.502314291113753 + 0.004134730741837304j),
                    (-2.308309109724023 + 1.733566216582246j),
                    (-0.2583568441893546 + 0.17747544422289296j),
                ],
            ],
        ),
        (
            "L(2000)",
            formulate_breit_wigner,
            [
                [
                    (0.5810314284422576 + 0.05238366127410415j),
                    (1.125156602679398 + 0.45145995736852845j),
                    (-1.3312729961106071 + 1.465055020402812j),
                ],
                [
                    (0.7185693849654539 + 0.06478357868676365j),
                    (1.3194116585653368 + 0.5294032223683035j),
                    (-1.039383287215348 + 1.1438327882458659j),
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
        func = create_function(
            expr.doit().xreplace(parameters | parameter_defaults),
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

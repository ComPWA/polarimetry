from __future__ import annotations

from os.path import dirname

from polarimetry.lhcb import load_model_builder, load_model_parameters
from polarimetry.lhcb.particle import load_particles

THIS_DIR = dirname(__file__)
DATA_DIR = f"{THIS_DIR}/../../data"


def _load_parameters(model_choice: int | str) -> dict[str, complex | float | int]:
    model_file = f"{DATA_DIR}/model-definitions.yaml"
    particles = load_particles(f"{DATA_DIR}/particle-definitions.yaml")
    model_builder = load_model_builder(model_file, particles, model_choice)
    symbol_parameters = load_model_parameters(
        filename=model_file,
        decay=model_builder.decay,
        model_id=model_choice,
    )
    parameters = {str(par): value for par, value in symbol_parameters.items()}
    print()
    print("Run pytest -s to view these parameter names:")
    print("  ", "\n  ".join(sorted(parameters)))
    return parameters


def test_load_model_parameters():
    parameters = _load_parameters("Default amplitude model")
    assert len(parameters) == 53
    assert parameters[R"\mathcal{H}^\mathrm{production}[K(892), 0, -1/2]"] == 1
    assert parameters[R"\gamma_{K(700)}"] == 0.94106
    assert (
        parameters[R"\mathcal{H}^\mathrm{production}[K(700), 0, 1/2]"]
        == 0.068908 + 2.521444j  # conversion factor +1
    )
    assert (
        parameters[R"\mathcal{H}^\mathrm{production}[K(700), 0, -1/2]"]
        == -2.685630 + 0.038490j  # conversion factor +1
    )

    parameters = _load_parameters(
        "Alternative amplitude model with L(1810) contribution added with free mass and"
        " width"
    )
    assert len(parameters) == 59
    assert parameters[R"\mathcal{H}^\mathrm{production}[K(892), 0, -1/2]"] == 1
    assert parameters[R"\gamma_{K(700)}"] == 0.857489
    assert (
        parameters[R"\mathcal{H}^\mathrm{production}[L(1810), -1/2, 0]"]
        == 0.865366 - 4.993321j  # conversion factor -1
    )
    assert (
        parameters[R"\mathcal{H}^\mathrm{production}[L(1810), 1/2, 0]"]
        == -1.179995 - 4.413438j  # conversion factor -1
    )

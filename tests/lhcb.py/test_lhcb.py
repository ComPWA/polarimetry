from __future__ import annotations

from os.path import dirname

import sympy as sp
from sympy.core.symbol import Str

from polarimetry.amplitude import DalitzPlotDecompositionBuilder
from polarimetry.lhcb import (
    get_conversion_factor_ls,
    load_model_builder,
    load_model_parameters,
)
from polarimetry.lhcb.particle import load_particles

THIS_DIR = dirname(__file__)
DATA_DIR = f"{THIS_DIR}/../../data"
MODEL_FILE = f"{DATA_DIR}/model-definitions.yaml"


def test_get_conversion_factor_ls():
    builder = _load_builder("Alternative amplitude model obtained using LS couplings")
    decay = builder.decay
    items = [
        f"{c.resonance.name:9s}"
        f"L={c.incoming_ls.L!r:3s}"
        f"S={c.incoming_ls.S!r:5s}"
        f"factor={get_conversion_factor_ls(c.decay):+d}"
        for c in decay.chains
    ]
    assert items == [
        "L(1405)  L=0  S=1/2  factor=+1",
        "L(1405)  L=1  S=1/2  factor=-1",
        "L(1520)  L=1  S=3/2  factor=-1",
        "L(1520)  L=2  S=3/2  factor=+1",
        "L(1600)  L=0  S=1/2  factor=-1",
        "L(1600)  L=1  S=1/2  factor=+1",
        "L(1670)  L=0  S=1/2  factor=+1",
        "L(1670)  L=1  S=1/2  factor=-1",
        "L(1690)  L=1  S=3/2  factor=-1",
        "L(1690)  L=2  S=3/2  factor=+1",
        "L(2000)  L=0  S=1/2  factor=+1",
        "L(2000)  L=1  S=1/2  factor=-1",
        "D(1232)  L=1  S=3/2  factor=-1",
        "D(1232)  L=2  S=3/2  factor=+1",
        "D(1600)  L=1  S=3/2  factor=-1",
        "D(1600)  L=2  S=3/2  factor=+1",
        "D(1700)  L=1  S=3/2  factor=+1",
        "D(1700)  L=2  S=3/2  factor=-1",
        "K(700)   L=0  S=1/2  factor=+1",
        "K(700)   L=1  S=1/2  factor=+1",
        "K(892)   L=0  S=1/2  factor=+1",
        "K(892)   L=1  S=1/2  factor=+1",
        "K(892)   L=1  S=3/2  factor=+1",
        "K(892)   L=2  S=3/2  factor=+1",
        "K(1430)  L=0  S=1/2  factor=+1",
        "K(1430)  L=1  S=1/2  factor=+1",
    ]


def test_load_model_parameters():
    parameters = _load_parameters("Default amplitude model")
    H_prod = sp.IndexedBase(R"\mathcal{H}^\mathrm{production}")
    gamma = sp.Symbol(R"\gamma_{K(700)}")
    h = sp.Rational(1, 2)
    assert len(parameters) == 53
    assert parameters[gamma] == 0.94106
    assert parameters[H_prod[Str("K(892)"), 0, -h]] == (1) * +1
    assert parameters[H_prod[Str("K(700)"), 0, +h]] == (0.068908 + 2.521444j) * +1
    assert parameters[H_prod[Str("K(700)"), 0, -h]] == (-2.685630 + 0.038490j) * +1

    parameters = _load_parameters(
        "Alternative amplitude model with L(1810) contribution added with free mass and"
        " width"
    )
    assert len(parameters) == 59
    assert parameters[gamma] == 0.857489
    assert parameters[H_prod[Str("K(892)"), 0, -h]] == (1) * +1
    assert parameters[H_prod[Str("L(1810)"), -h, 0]] == (-0.865366 + 4.993321j) * -1
    assert parameters[H_prod[Str("L(1810)"), +h, 0]] == (1.179995 + 4.413438j) * -1


def _load_builder(model_choice: int | str) -> DalitzPlotDecompositionBuilder:
    particles = load_particles(f"{DATA_DIR}/particle-definitions.yaml")
    return load_model_builder(MODEL_FILE, particles, model_choice)


def _load_parameters(
    model_choice: int | str,
) -> dict[sp.Indexed | sp.Symbol, complex | float]:
    model_builder = _load_builder(model_choice)
    return load_model_parameters(
        filename=MODEL_FILE,
        decay=model_builder.decay,
        model_id=model_choice,
    )

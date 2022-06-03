# cspell:ignore modelparameters modelstudies
"""Import functions that are specifically for this LHCb analysis.

.. seealso:: :doc:`/cross-check`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import sympy as sp
from sympy.core.symbol import Str

from polarization.decay import IsobarNode, Particle, ThreeBodyDecay, ThreeBodyDecayChain
from polarization.spin import filter_parity_violating_ls, generate_ls_couplings

if sys.version_info < (3, 8):
    from typing_extensions import Literal, TypedDict
else:
    from typing import Literal, TypedDict


Λc = Particle(
    name="Λc⁺",
    latex=R"\Lambda_c^+",
    spin=0.5,
    parity=+1,
    mass=2.28646,
    width=3.25e-12,
)
p = Particle(
    name="p",
    latex="p",
    spin=0.5,
    parity=+1,
    mass=0.938272046,
    width=0.0,
)
K = Particle(
    name="K⁻",
    latex="K^-",
    spin=0,
    parity=-1,
    mass=0.493677,
    width=5.317e-17,
)
π = Particle(
    name="π⁺",
    latex=R"\pi^+",
    spin=0,
    parity=-1,
    mass=0.13957018,
    width=2.5284e-17,
)

# https://github.com/redeboer/polarization-sensitivity/blob/34f5330/julia/notebooks/model0.jl#L43-L47
Σ = Particle(
    name="Σ⁻",
    latex=R"\Sigma^-",
    spin=0.5,
    parity=+1,
    mass=1.18937,
    width=4.45e-15,
)


def load_three_body_decays(filename: str) -> ThreeBodyDecay:
    def create_isobar(resonance: Particle) -> ThreeBodyDecayChain:
        if resonance.name.startswith("K"):
            child1, child2, sibling = π, K, p
        elif resonance.name.startswith("L"):
            child1, child2, sibling = K, p, π
        elif resonance.name.startswith("D"):
            child1, child2, sibling = p, π, K
        else:
            raise NotImplementedError
        decay = IsobarNode(
            parent=Λc,
            child1=sibling,
            child2=IsobarNode(
                parent=resonance,
                child1=child1,
                child2=child2,
                interaction=generate_L_min(
                    resonance, child1, child2, conserve_parity=True
                ),
            ),
            interaction=generate_L_min(Λc, sibling, resonance, conserve_parity=False),
        )
        return ThreeBodyDecayChain(decay)

    def generate_L_min(
        parent: Particle, child1: Particle, child2: Particle, conserve_parity: bool
    ) -> int:
        ls = generate_ls_couplings(parent.spin, child1.spin, child2.spin)
        if conserve_parity:
            ls = filter_parity_violating_ls(
                ls, parent.parity, child1.parity, child2.parity
            )
        return min(ls)

    resonances = load_resonance_definitions(filename)
    return ThreeBodyDecay(
        states={
            0: Λc,
            1: p,
            2: π,
            3: K,
        },
        chains=tuple(create_isobar(res) for res in resonances.values()),
    )


def load_resonance_definitions(filename: Path | str) -> dict[str, Particle]:
    """Load `Particle` definitions from a JSON file."""
    with open(filename) as stream:
        data = json.load(stream)
    isobar_definitions = data["isobars"]
    return to_resonance_dict(isobar_definitions)


def load_model_parameters(
    filename: Path | str, model_number: int = 0
) -> dict[sp.Indexed | sp.Symbol, complex | float]:
    with open("../data/modelparameters.json") as stream:
        json_data = json.load(stream)
    json_parameters = json_data["modelstudies"][model_number]["parameters"]
    json_parameters["ArK(892)1"] = "1.0 ± 0.0"
    json_parameters["AiK(892)1"] = "0.0 ± 0.0"
    return to_symbol_definitions(json_parameters)


def to_resonance_dict(definition: dict[str, ResonanceJSON]) -> dict[str, Particle]:
    return {
        name: to_resonance(name, resonance_def)
        for name, resonance_def in definition.items()
    }


def to_resonance(name: str, definition: ResonanceJSON) -> Particle:
    spin, parity = _to_jp_pair(definition["jp"])
    return Particle(
        name,
        name,
        spin,
        parity,
        mass=_average_float(definition["mass"]) * 1e-3,  # MeV to GeV
        width=_average_float(definition["width"]) * 1e-3,  # MeV to GeV
        lineshape=definition["lineshape"],
    )


def _average_float(input_str: str) -> tuple[float, float]:
    """
    >>> _average_float("1405.1")
    1405.1
    >>> _average_float("1900-2100")
    2000.0
    """
    if "-" in input_str:
        _min, _max, *_ = map(float, input_str.split("-"))
        return (_max + _min) / 2
    return float(input_str)


def _to_jp_pair(input_str: str) -> tuple[sp.Rational, int]:
    """
    >>> _to_jp_pair("3/2^-")
    (3/2, -1)
    >>> _to_jp_pair("0^+")
    (0, 1)
    """
    spin, parity_sign = input_str.split("^")
    return sp.Rational(spin), int(f"{parity_sign}1")


class ResonanceJSON(TypedDict):
    jp: str
    mass: str
    width: str
    lineshape: Literal["BreitWignerMinL", "BuggBreitWignerMinL", "Flatte1405"]


def to_symbol_definitions(
    parameter_dict: dict[str, str]
) -> dict[sp.Basic, complex | float]:
    key_to_val: dict[str, complex | float] = {}
    for key, str_value in parameter_dict.items():
        if key.startswith("Ar"):
            identifier = key[2:]
            str_imag = parameter_dict[f"Ai{identifier}"]
            real = to_float(str_value)
            imag = to_float(str_imag)
            key_to_val[f"A{identifier}"] = complex(real, imag)
        elif key.startswith("Ai"):
            continue
        else:
            key_to_val[key] = to_float(str_value)
    return {to_symbol(key): value for key, value in key_to_val.items()}


def to_float(str_value: str) -> float:
    value, _ = map(float, str_value.split(" ± "))
    return value


def to_symbol(key: str) -> sp.Indexed | sp.Symbol:
    H_prod = sp.IndexedBase(R"\mathcal{H}^\mathrm{production}")
    half: sp.Rational = sp.S.Half
    if key.startswith("A"):
        res = stringify(key[1:-1])
        i = int(key[-1])
        if str(res).startswith("L"):
            if i == 1:
                return H_prod[res, +half, 0]
            if i == 2:
                return H_prod[res, -half, 0]
        if str(res).startswith("D"):
            if i == 1:
                return H_prod[res, +half, 0]
            if i == 2:
                return H_prod[res, -half, 0]
        if str(res).startswith("K"):
            if str(res) in {"K(700)", "K(1430)"}:
                if i == 1:
                    return H_prod[res, 0, -half]
                if i == 2:
                    return H_prod[res, 0, +half]
            else:
                if i == 1:
                    return H_prod[res, 0, +half]
                if i == 2:
                    return H_prod[res, -1, +half]
                if i == 3:
                    return H_prod[res, +1, -half]
                if i == 4:
                    return H_prod[res, 0, -half]
    if key.startswith("gamma"):
        res = stringify(key[5:])
        return sp.Symbol(Rf"\gamma_{{{res}}}")
    if key.startswith("M"):
        res = stringify(key[1:])
        return sp.Symbol(Rf"m_{{{res}}}")
    if key.startswith("G"):
        res = stringify(key[1:])
        return sp.Symbol(Rf"\Gamma_{{{res}}}")
    raise NotImplementedError(
        f'Cannot convert key "{key}" in model parameter JSON file to SymPy symbol'
    )


def stringify(obj) -> Str:
    if isinstance(obj, Particle):
        return Str(obj.latex)
    return Str(f"{obj}")

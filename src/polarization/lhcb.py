"""Import functions that are specifically for this LHCb analysis."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import sympy as sp

from polarization.decay import IsobarNode, Resonance, ThreeBodyDecay
from polarization.spin import filter_parity_violating_ls, generate_ls_couplings

if sys.version_info < (3, 8):
    from typing_extensions import Literal, TypedDict
else:
    from typing import Literal, TypedDict


Λc = Resonance(
    name="Λc⁺",
    latex=R"\Lambda_c^+",
    spin=0.5,
    parity=+1,
    mass=2.28646,
    width=3.25e-12,
)
p = Resonance(
    name="p",
    latex="p",
    spin=0.5,
    parity=+1,
    mass=0.938272046,
    width=0.0,
)
K = Resonance(
    name="K⁻",
    latex="K^-",
    spin=0,
    parity=-1,
    mass=0.493677,
    width=5.317e-17,
)
π = Resonance(
    name="π⁺",
    latex=R"\pi^+",
    spin=0,
    parity=-1,
    mass=0.13957018,
    width=2.5284e-17,
)

# https://github.com/redeboer/polarization-sensitivity/blob/34f5330/julia/notebooks/model0.jl#L43-L47
Σ = Resonance(
    name="Σ⁻",
    latex=R"\Sigma^-",
    spin=0.5,
    parity=+1,
    mass=1.18937,
    width=4.45e-15,
)


def load_three_body_decays(filename: str) -> ThreeBodyDecay:
    def create_isobar(resonance: Resonance) -> ThreeBodyDecay:
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
        return ThreeBodyDecay(decay)

    def generate_L_min(
        parent: Resonance, child1: Resonance, child2: Resonance, conserve_parity: bool
    ) -> int:
        ls = generate_ls_couplings(parent.spin, child1.spin, child2.spin)
        if conserve_parity:
            ls = filter_parity_violating_ls(
                ls, parent.parity, child1.parity, child2.parity
            )
        return min(ls)

    resonances = load_resonance_definitions(filename)
    return [create_isobar(res) for res in resonances.values()]


def load_resonance_definitions(filename: Path | str) -> dict[str, Resonance]:
    """Load `Resonance` definitions from a JSON file."""
    with open(filename) as stream:
        data = json.load(stream)
    isobar_definitions = data["isobars"]
    return to_resonance_dict(isobar_definitions)


def to_resonance_dict(definition: dict[str, ResonanceJSON]) -> dict[str, Resonance]:
    return {
        name: to_resonance(name, resonance_def)
        for name, resonance_def in definition.items()
    }


def to_resonance(name: str, definition: ResonanceJSON) -> Resonance:
    spin, parity = _to_jp_pair(definition["jp"])
    return Resonance(
        name,
        name,
        spin,
        parity,
        mass=_average_float(definition["mass"], factor=1e-3),  # MeV to GeV
        width=_average_float(definition["width"], factor=1e-3),  # MeV to GeV
        lineshape=definition["lineshape"],
    )


def _average_float(input_str: str, factor: float = 1) -> tuple[float, float]:
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

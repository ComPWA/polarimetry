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


def load_three_body_decays(
    filename: str,
    Λc: Resonance,
    p: Resonance,
    π: Resonance,
    K: Resonance,
) -> ThreeBodyDecay:
    def create_isobar(resonance: Resonance) -> ThreeBodyDecay:
        if resonance.name.startswith("K"):
            child1, child2, sibling = π, K, p
        elif resonance.name.startswith("L"):
            child1, child2, sibling = p, K, π
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
                interaction=generate_L_min(resonance, child1, child2),
            ),
            interaction=generate_L_min(Λc, sibling, resonance),
        )
        return ThreeBodyDecay(decay)

    def generate_L_min(parent: Resonance, child1: Resonance, child2: Resonance) -> int:
        ls = generate_ls_couplings(parent.spin, child1.spin, child2.spin)
        ls = filter_parity_violating_ls(ls, parent.parity, child1.parity, child2.parity)
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
        mass_range=_to_float_range(definition["mass"], factor=1e-3),  # MeV to GeV
        width_range=_to_float_range(definition["width"], factor=1e-3),  # MeV to GeV
        lineshape=definition["lineshape"],
    )


def _to_float_range(input_str: str, factor: float = 1) -> tuple[float, float]:
    """
    >>> _to_float_range("1405.1")
    (1405.1, 1405.1)
    >>> _to_float_range("1900-2100")
    (1900.0, 2100.0)
    """
    if "-" in input_str:
        _min, _max, *_ = map(float, input_str.split("-"))
    else:
        _min = _max = float(input_str)
    return (
        _min * factor,
        _max * factor,
    )


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

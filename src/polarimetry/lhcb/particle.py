"""Hard-coded particle definitions."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import sympy as sp
import yaml

from polarimetry.decay import Particle


def load_particles(filename: Path | str) -> dict[str, Particle]:
    """Load `.Particle` definitions from a YAML file."""
    particle_definitions = _load_particles_json(filename)
    return _to_resonance_dict(particle_definitions)


def _load_particles_json(filename: Path | str) -> dict[str, ResonanceJSON]:
    with open(filename) as stream:
        return yaml.load(stream, Loader=yaml.SafeLoader)


def _to_resonance_dict(definition: dict[str, ResonanceJSON]) -> dict[str, Particle]:
    return {
        name: _to_resonance(name, resonance_def)
        for name, resonance_def in definition.items()
    }


def _to_resonance(name: str, definition: ResonanceJSON) -> Particle:
    spin, parity = _to_jp_pair(definition["jp"])
    latex = definition.get("latex", name)
    return Particle(
        name,
        latex,
        spin,
        parity,
        mass=_average_float(definition["mass"]) * 1e-3,  # MeV to GeV
        width=_average_float(definition["width"]) * 1e-3,  # MeV to GeV
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


def _average_float(input_str: float | str) -> tuple[float, float]:
    """
    >>> _average_float("1405.1")
    1405.1
    >>> _average_float("1900-2100")
    2000.0
    """
    if isinstance(input_str, str) and "-" in input_str:
        _min, _max, *_ = map(float, input_str.split("-"))
        return (_max + _min) / 2
    return float(input_str)


class ResonanceJSON(TypedDict):
    latex: str
    jp: str
    mass: float | str
    width: float | str


__PARTICLE_DATABASE = load_particles(
    Path(__file__).parent.parent / "lhcb/particle-definitions.yaml"
)

Λc = __PARTICLE_DATABASE["Lambda_c+"]
p = __PARTICLE_DATABASE["p"]
K = __PARTICLE_DATABASE["K-"]
π = __PARTICLE_DATABASE["pi+"]
PARTICLE_TO_ID = {Λc: 0, p: 1, π: 2, K: 3}

# https://github.com/ComPWA/polarimetry/blob/34f5330/julia/notebooks/model0.jl#L43-L47
Σ = __PARTICLE_DATABASE["Sigma-"]

"""Data structures that describe a three-body decay."""
from __future__ import annotations

import sys

import sympy as sp
from attrs import field, frozen

from polarization._attrs import assert_spin_value, to_ls, to_rational

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal


@frozen
class ThreeBodyDecay:
    initial_state: Particle
    final_state: tuple[Particle, Particle, Particle]
    resonances: tuple[IsobarNode, ...]

    def __attrs_post_init__(self) -> None:
        for resonance in self.resonances:
            if self.final_state != resonance.children:
                final_state = ", ".join(p.name for p in self.final_state)
                raise ValueError(
                    f"Resonance {resonance.parent.name} →"
                    f" {resonance.child1.name} {resonance.child2.name} does not decay"
                    f" to {final_state}"
                )


@frozen
class Particle:
    name: str
    latex: str
    spin: sp.Rational = field(converter=to_rational, validator=assert_spin_value)
    parity: int


@frozen
class Resonance(Particle):
    mass_range: tuple[float, float]
    width_range: tuple[float, float]
    lineshape: Literal["BreitWignerMinL", "BuggBreitWignerMinL", "Flatte1405"]

    @property
    def mass(self) -> float:
        return _compute_average(self.mass_range)

    @property
    def width(self) -> float:
        return _compute_average(self.width_range)


def _compute_average(range_def: float | tuple[float, float]) -> float:
    _min, _max = range_def
    return (_max + _min) / 2


@frozen
class IsobarNode:
    parent: Resonance
    child1: Resonance | IsobarNode
    child2: Resonance | IsobarNode
    interaction: LSCoupling | None = field(default=None, converter=to_ls)

    @property
    def children(self) -> tuple[Particle, Particle]:
        return self.child1, self.child2


@frozen
class LSCoupling:
    L: int
    S: sp.Rational = field(converter=to_rational, validator=assert_spin_value)

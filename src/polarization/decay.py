"""Data structures that describe a three-body decay."""
from __future__ import annotations

import sympy as sp
from attrs import field, frozen
from attrs.validators import instance_of

from polarization._attrs import assert_spin_value, to_ls, to_rational


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
    spin: sp.Rational = field(converter=to_rational, validator=assert_spin_value)
    parity: int


@frozen
class IsobarNode:
    parent: Particle = field(validator=instance_of(Particle))
    child1: Particle = field(validator=instance_of(Particle))
    child2: Particle = field(validator=instance_of(Particle))
    interaction: LSCoupling | None = field(default=None, converter=to_ls)

    @property
    def children(self) -> tuple[Particle, Particle]:
        return self.child1, self.child2


@frozen
class LSCoupling:
    L: int
    S: sp.Rational = field(converter=to_rational, validator=assert_spin_value)

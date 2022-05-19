"""Data structures that describe a three-body decay."""
from __future__ import annotations

import sys

import sympy as sp
from attrs import field, frozen
from attrs.validators import instance_of

from polarization._attrs import assert_spin_value, to_ls, to_rational

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal


@frozen
class Particle:
    name: str
    latex: str
    spin: sp.Rational = field(converter=to_rational, validator=assert_spin_value)
    parity: int


@frozen
class Resonance(Particle):
    mass: float
    width: float
    lineshape: Literal[
        "BreitWignerMinL", "BuggBreitWignerMinL", "Flatte1405"
    ] | None = None


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
class ThreeBodyDecay:
    decay: IsobarNode = field(validator=instance_of(IsobarNode))

    def __attrs_post_init__(self) -> None:
        if not isinstance(self.decay.child1, Resonance):
            raise TypeError(f"Child 1 has of type {Resonance.__name__} (spectator)")
        if not isinstance(self.decay.child2, IsobarNode):
            raise TypeError(f"Child 2 has of type {IsobarNode.__name__} (the decay)")
        if not isinstance(self.decay.child2.child1, Resonance):
            raise TypeError(f"Child 1 of child 2 has of type {Resonance.__name__}")
        if not isinstance(self.decay.child2.child1, Resonance):
            raise TypeError(f"Child 1 of child 2 has of type {Resonance.__name__}")
        if self.incoming_ls is None:
            raise ValueError(f"LS-coupling for production node required")
        if self.outgoing_ls is None:
            raise ValueError(f"LS-coupling for decay node required")

    @property
    def parent(self) -> Resonance:
        return self.decay.parent

    @property
    def spectator(self) -> Resonance:
        return self.decay.child1

    @property
    def resonance(self) -> Resonance:
        return self.decay.child2.parent

    @property
    def decay_products(self) -> tuple[Resonance, Resonance]:
        return (
            self.decay.child2.child1,
            self.decay.child2.child2,
        )

    @property
    def incoming_ls(self) -> LSCoupling:
        return self.decay.interaction

    @property
    def outgoing_ls(self) -> LSCoupling:
        return self.decay.child2.interaction


@frozen
class LSCoupling:
    L: int
    S: sp.Rational = field(converter=to_rational, validator=assert_spin_value)

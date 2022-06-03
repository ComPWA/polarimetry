"""Data structures that describe a three-body decay."""
from __future__ import annotations

import sys
from typing import Dict

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
    mass: float
    width: float
    lineshape: Literal[
        "BreitWignerMinL", "BuggBreitWignerMinL", "Flatte1405"
    ] | None = None


@frozen
class IsobarNode:
    parent: Particle
    child1: Particle | IsobarNode
    child2: Particle | IsobarNode
    interaction: LSCoupling | None = field(default=None, converter=to_ls)

    @property
    def children(self) -> tuple[Particle, Particle]:
        return self.child1, self.child2


@frozen
class ThreeBodyDecay:
    states: StateDict
    chains: tuple[ThreeBodyDecayChain, ...]

    def __attrs_post_init__(self) -> None:
        expected_initial_state = self.initial_state
        expected_final_state = set(self.final_state.values())
        for i, chain in enumerate(self.chains):
            if chain.parent != expected_initial_state:
                raise ValueError(
                    f"Chain {i} has initial state {chain.parent.name}, but should have"
                    f" {expected_initial_state.name}"
                )
            final_state = {chain.spectator, *chain.decay_products}
            if final_state != expected_final_state:
                to_str = lambda s: ", ".join(p.name for p in s)
                raise ValueError(
                    f"Chain {i} has final state {to_str(final_state)}, but should have"
                    f" {to_str(expected_final_state)}"
                )

    @property
    def initial_state(self) -> Particle:
        return self.states[0]

    @property
    def final_state(self) -> dict[Literal[1, 2, 3], Particle]:
        return {k: v for k, v in self.states.items() if k != 0}

    def find_chain(self, resonance_name: str) -> ThreeBodyDecayChain:
        for chain in self.chains:
            if chain.resonance.name == resonance_name:
                return chain
        raise KeyError(f"No decay chain found for resonance {resonance_name}")


StateDict = Dict[Literal[0, 1, 2, 3], Particle]


@frozen
class ThreeBodyDecayChain:
    decay: IsobarNode = field(validator=instance_of(IsobarNode))

    def __attrs_post_init__(self) -> None:
        if not isinstance(self.decay.child1, Particle):
            raise TypeError(f"Child 1 has of type {Particle.__name__} (spectator)")
        if not isinstance(self.decay.child2, IsobarNode):
            raise TypeError(f"Child 2 has of type {IsobarNode.__name__} (the decay)")
        if not isinstance(self.decay.child2.child1, Particle):
            raise TypeError(f"Child 1 of child 2 has of type {Particle.__name__}")
        if not isinstance(self.decay.child2.child1, Particle):
            raise TypeError(f"Child 1 of child 2 has of type {Particle.__name__}")
        if self.incoming_ls is None:
            raise ValueError(f"LS-coupling for production node required")
        if self.outgoing_ls is None:
            raise ValueError(f"LS-coupling for decay node required")

    @property
    def parent(self) -> Particle:
        return self.decay.parent

    @property
    def spectator(self) -> Particle:
        return self.decay.child1

    @property
    def resonance(self) -> Particle:
        return self.decay.child2.parent

    @property
    def decay_products(self) -> tuple[Particle, Particle]:
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

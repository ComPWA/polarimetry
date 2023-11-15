"""Data structures that describe a three-body decay."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Literal

from attrs import field, frozen
from attrs.validators import instance_of

from polarimetry._attrs import assert_spin_value, to_ls, to_rational

if TYPE_CHECKING:
    import sympy as sp


@frozen
class Particle:
    name: str
    latex: str
    spin: sp.Rational = field(converter=to_rational, validator=assert_spin_value)
    parity: Literal[-1, 1]
    mass: float
    width: float


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
    states: OuterStates
    chains: tuple[ThreeBodyDecayChain, ...]

    def __attrs_post_init__(self) -> None:
        expected_initial_state = self.initial_state
        expected_final_state = set(self.final_state.values())
        for i, chain in enumerate(self.chains):
            if chain.parent != expected_initial_state:
                msg = (
                    f"Chain {i} has initial state {chain.parent.name}, but should have"
                    f" {expected_initial_state.name}"
                )
                raise ValueError(msg)
            final_state = {chain.spectator, *chain.decay_products}
            if final_state != expected_final_state:
                to_str = lambda s: ", ".join(p.name for p in s)  # noqa: E731
                msg = (
                    f"Chain {i} has final state {to_str(final_state)}, but should have"
                    f" {to_str(expected_final_state)}"
                )
                raise ValueError(msg)

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
        msg = f"No decay chain found for resonance {resonance_name}"
        raise KeyError(msg)

    def get_subsystem(self, subsystem_id: Literal[1, 2, 3]) -> ThreeBodyDecay:
        child1_id, child2_id = get_decay_product_ids(subsystem_id)
        child1 = self.final_state[child1_id]
        child2 = self.final_state[child2_id]
        filtered_chains = [
            chain
            for chain in self.chains
            if chain.decay_products in {(child1, child2), (child2, child1)}
        ]
        return ThreeBodyDecay(self.states, filtered_chains)


def get_decay_product_ids(spectator_id: Literal[1, 2, 3]) -> tuple[int, int]:
    if spectator_id == 1:
        return 2, 3
    if spectator_id == 2:
        return 3, 1
    if spectator_id == 3:
        return 1, 2
    msg = f"Spectator ID has to be one of 1, 2, 3, not {spectator_id}"
    raise ValueError(msg)


OuterStates = Dict[Literal[0, 1, 2, 3], Particle]
"""Mapping of the initial and final state IDs to their `.Particle` definition."""


@frozen
class ThreeBodyDecayChain:
    decay: IsobarNode = field(validator=instance_of(IsobarNode))

    def __attrs_post_init__(self) -> None:
        if not isinstance(self.decay.child1, IsobarNode):
            msg = f"Child 1 has of type {IsobarNode.__name__} (the decay)"
            raise TypeError(msg)
        if not isinstance(self.decay.child1.child1, Particle):
            msg = f"Child 1 of child 1 has of type {Particle.__name__}"
            raise TypeError(msg)
        if not isinstance(self.decay.child1.child1, Particle):
            msg = f"Child 1 of child 1 has of type {Particle.__name__}"
            raise TypeError(msg)
        if not isinstance(self.decay.child2, Particle):
            msg = f"Child 2 has of type {Particle.__name__} (spectator)"
            raise TypeError(msg)
        if self.incoming_ls is None:  # pyright: ignore[reportUnnecessaryComparison]
            msg = "LS-coupling for production node required"
            raise ValueError(msg)
        if self.outgoing_ls is None:  # pyright: ignore[reportUnnecessaryComparison]
            msg = "LS-coupling for decay node required"
            raise ValueError(msg)

    @property
    def parent(self) -> Particle:
        return self.decay.parent

    @property
    def resonance(self) -> Particle:
        return self.decay.child1.parent

    @property
    def decay_products(self) -> tuple[Particle, Particle]:
        return (
            self.decay.child1.child1,
            self.decay.child1.child2,
        )

    @property
    def spectator(self) -> Particle:
        return self.decay.child2

    @property
    def incoming_ls(self) -> LSCoupling:
        return self.decay.interaction

    @property
    def outgoing_ls(self) -> LSCoupling:
        return self.decay.child1.interaction


@frozen
class LSCoupling:
    L: int
    S: sp.Rational = field(converter=to_rational, validator=assert_spin_value)

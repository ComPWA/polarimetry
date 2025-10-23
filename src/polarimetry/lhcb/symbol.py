"""Create symbols with the correct assumptions for LHCb polarimetry models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

import sympy as sp
from ampform_dpd.decay import State

if TYPE_CHECKING:
    from typing import Literal


class Particle(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def latex(self) -> str: ...


def create_mass_symbol(particle: Particle | State | str) -> sp.Symbol:
    if isinstance(particle, State):
        return sp.Symbol(f"m{particle.index}", nonnegative=True)
    particle = _get_latex(particle)
    return sp.Symbol(f"m_{{{particle}}}", nonnegative=True)


def create_width_symbol(
    particle: str | Particle,
    decay_products: tuple[str | Particle, str | Particle] | None = None,
) -> sp.Symbol:
    particle = _get_latex(particle)
    suffix = ""
    if decay_products:
        p1, p2 = map(_get_latex, decay_products)
        suffix = Rf" \to {p1} {p2}"
    return sp.Symbol(Rf"\Gamma_{{{particle}{suffix}}}", nonnegative=True)


def create_meson_radius_symbol(subscript: Literal["prod", "dec"]) -> sp.Symbol:
    return sp.Symbol(Rf"R_\mathrm{{{subscript}}}", positive=True)


def create_alpha_symbol(particle: str | Particle) -> sp.Symbol:
    """Define symbol :math:`alpha` for `this paper <https://arxiv.org/pdf/hep-ex/0510019.pdf>`_."""
    particle = _get_latex(particle)
    return sp.Symbol(Rf"\alpha_{{{particle}}}", real=True)


def create_gamma_symbol(particle: str | Particle) -> sp.Symbol:
    """Define symbol :math:`gamma` for `this paper <https://arxiv.org/pdf/hep-ex/0510019.pdf>`_."""
    particle = _get_latex(particle)
    return sp.Symbol(Rf"\gamma_{{{particle}}}", nonnegative=True)


def _get_latex(particle: str | Particle) -> str:
    if latex := getattr(particle, "latex", None):
        return latex
    return str(particle)

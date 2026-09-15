"""Dynamics lineshape definitions for the LHCb amplitude model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sympy as sp
from ampform.dynamics.form_factor import FormFactor
from ampform.dynamics.phasespace import BreakupMomentum
from ampform.sympy import unevaluated
from ampform_dpd import DefinedExpression
from ampform_dpd.dynamics import (
    BreitWignerBuilder,
    _get_angular_momentum,  # ruff: ignore[import-private-name]
    get_mandelstam_s,
)

from polarimetry.lhcb.particle import Σ, π
from polarimetry.lhcb.symbol import (
    create_alpha_symbol,
    create_gamma_symbol,
    create_mass_symbol,
    create_meson_radius_symbol,
    create_width_symbol,
)

if TYPE_CHECKING:
    from ampform_dpd.decay import ThreeBodyDecayChain


@unevaluated
class BuggBreitWigner(sp.Expr):
    s: Any
    m0: Any
    Γ0: Any
    m1: Any
    m2: Any
    γ: Any
    _latex_repr_ = R"\mathcal{{R}}^\mathrm{{Bugg}}\left({s}\right)"

    def evaluate(self):
        s, m0, Γ0, m1, m2, γ = self.args
        # Adler zero
        s_A = m1**2 - m2**2 / 2  # ty: ignore[unsupported-operator]
        g_squared = sp.Mul(
            (s - s_A) / (m0**2 - s_A),  # ty: ignore[unsupported-operator]
            m0 * Γ0 * sp.exp(-γ * s),  # ty: ignore[unsupported-operator]
            evaluate=False,
        )
        return 1 / (m0**2 - s - sp.I * g_squared)  # ty: ignore[unsupported-operator]


@unevaluated
class FlattéSWave(sp.Expr):
    # https://github.com/ComPWA/polarimetry/blob/34f5330/julia/notebooks/model0.jl#L151-L161
    s: Any
    m0: Any
    widths: tuple[Any, Any]
    masses1: tuple[Any, Any]
    masses2: tuple[Any, Any]
    _latex_repr_ = R"\mathcal{{R}}^\mathrm{{Flatté}}\left({s}\right)"

    def evaluate(self):
        m0: sp.Expr
        s, m0, (Γ1, Γ2), (ma1, mb1), (ma2, mb2) = self.args  # ty: ignore[not-iterable, invalid-assignment]
        p = BreakupMomentum(s, ma1, mb1)
        p0 = BreakupMomentum(m0**2, ma2, mb2)
        q = BreakupMomentum(s, ma2, mb2)
        q0 = BreakupMomentum(m0**2, ma2, mb2)
        Γ1 *= (p / p0) * m0 / sp.sqrt(s)
        Γ2 *= (q / q0) * m0 / sp.sqrt(s)
        Γ = Γ1 + Γ2
        return 1 / (m0**2 - s - sp.I * m0 * Γ)


def formulate_bugg_breit_wigner(decay_chain: ThreeBodyDecayChain) -> DefinedExpression:
    if {s.index for s in decay_chain.decay_products} != {2, 3}:
        msg = f"Bugg Breit-Wigner only defined for K* → Kπ (subsystem 1, not {decay_chain.spectator.index})"
        raise ValueError(msg)
    s = get_mandelstam_s(decay_chain.decay_node)
    m2, m3 = _create_decay_product_masses(decay_chain)
    gamma = create_gamma_symbol(decay_chain.resonance)
    mass = create_mass_symbol(decay_chain.resonance)
    width = create_width_symbol(decay_chain.resonance)
    return DefinedExpression(
        expression=BuggBreitWigner(s, mass, width, m3, m2, gamma),
        parameters={
            mass: decay_chain.resonance.mass,
            width: decay_chain.resonance.width,
            m2: decay_chain.decay_products[0].mass,
            m3: decay_chain.decay_products[1].mass,
            gamma: 1,
        },
    )


def formulate_exponential_bugg_breit_wigner(
    decay_chain: ThreeBodyDecayChain,
) -> DefinedExpression:
    """See `this paper, Eq. (4) <https://arxiv.org/pdf/hep-ex/0510019.pdf#page=3>`_."""
    bugg_bw = formulate_bugg_breit_wigner(decay_chain)
    alpha = create_alpha_symbol(decay_chain.resonance)
    s = get_mandelstam_s(decay_chain.decay_node)
    m0, m1 = sp.symbols("m0 m1", nonnegative=True)
    q = BreakupMomentum(m0**2, sp.sqrt(s), m1)
    return DefinedExpression(
        expression=bugg_bw.expression * sp.exp(-alpha * q**2),
        parameters={**bugg_bw.parameters, alpha: sp.Rational(0)},  # ty:ignore[invalid-argument-type]
    )


def formulate_flatte_1405(  # ruff: ignore[too-many-locals]
    decay_chain: ThreeBodyDecayChain,
) -> DefinedExpression:
    s = get_mandelstam_s(decay_chain.decay_node)
    resonance = decay_chain.resonance
    p1, p2 = decay_chain.decay_products
    m1, m2 = _create_decay_product_masses(decay_chain)
    m_res = create_mass_symbol(resonance)
    Γ1 = create_width_symbol(resonance, (p1, p2))
    Γ2 = create_width_symbol(resonance, (π, Σ))
    m_top = create_mass_symbol(decay_chain.parent)
    m_spec = create_mass_symbol(decay_chain.spectator)
    mπ = create_mass_symbol(π)
    mΣ = create_mass_symbol(Σ)
    l_prod = _get_angular_momentum(decay_chain.production_node)
    R_prod = create_meson_radius_symbol("prod")
    ff = FormFactor(m_top**2, sp.sqrt(s), m_spec, l_prod, R_prod)  # ty:ignore[invalid-argument-type]
    ff0 = FormFactor(m_top**2, m_res, m_spec, l_prod, R_prod)  # ty:ignore[invalid-argument-type]
    return DefinedExpression(
        expression=ff / ff0 * FlattéSWave(s, m_res, (Γ1, Γ2), (m1, m2), (mπ, mΣ)),  # ty:ignore[invalid-argument-type]
        parameters={
            m_res: resonance.mass,
            Γ1: resonance.width,
            Γ2: resonance.width,
            m1: decay_chain.decay_products[0].mass,
            m2: decay_chain.decay_products[1].mass,
            m_top: decay_chain.parent.mass,
            m_spec: decay_chain.spectator.mass,
            mπ: π.mass,
            mΣ: Σ.mass,
            # https://github.com/ComPWA/polarimetry/pull/11#issuecomment-1128784376
            R_prod: 5,
        },
    )


def formulate_breit_wigner(decay_chain: ThreeBodyDecayChain) -> DefinedExpression:
    """Relativistic Breit-Wigner with pole-normalized production and decay vertices.

    This is the ``BreitWignerMinL`` lineshape of the published LHCb models, expressed
    through AmpForm-DPD's common `~ampform_dpd.dynamics.BreitWignerBuilder`.
    """
    R_dec = create_meson_radius_symbol("dec")
    R_prod = create_meson_radius_symbol("prod")
    builder = BreitWignerBuilder(
        normalize_form_factors=True,
        meson_radius=lambda isobar: (
            R_prod if isobar.parent == decay_chain.parent else R_dec
        ),
        # https://github.com/ComPWA/polarimetry/pull/11#issuecomment-1128784376
        parameter_defaults={R_dec: 1.5, R_prod: 5},
    )
    return builder(decay_chain)


def _create_decay_product_masses(
    decay_chain: ThreeBodyDecayChain,
) -> tuple[sp.Symbol, sp.Symbol]:
    mi, mj = (create_mass_symbol(p) for p in decay_chain.decay_products)
    return mi, mj

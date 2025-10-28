"""Dynamics lineshape definitions for the LHCb amplitude model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp
from ampform.dynamics.phasespace import BreakupMomentum
from ampform_dpd.dynamics import (
    BlattWeisskopf,
    BreitWignerMinL,
    BuggBreitWigner,
    FlattéSWave,
)
from ampform_dpd.dynamics.builder import (
    _get_angular_momentum,  # noqa: PLC2701  # pyright:ignore[reportPrivateUsage]
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


def formulate_bugg_breit_wigner(
    decay_chain: ThreeBodyDecayChain,
) -> tuple[BuggBreitWigner, dict[sp.Symbol, complex | float]]:
    if {s.index for s in decay_chain.decay_products} != {2, 3}:
        msg = f"Bugg Breit-Wigner only defined for K* → Kπ (subsystem 1, not {decay_chain.spectator.index})"
        raise ValueError(msg)
    s = get_mandelstam_s(decay_chain.decay_node)
    m2, m3 = _create_decay_product_masses(decay_chain)
    gamma = create_gamma_symbol(decay_chain.resonance)
    mass = create_mass_symbol(decay_chain.resonance)
    width = create_width_symbol(decay_chain.resonance)
    expression = BuggBreitWigner(s, mass, width, m3, m2, gamma)
    parameter_defaults: dict[sp.Symbol, complex | float] = {
        mass: decay_chain.resonance.mass,
        width: decay_chain.resonance.width,
        m2: decay_chain.decay_products[0].mass,
        m3: decay_chain.decay_products[1].mass,
        gamma: 1,
    }
    return expression, parameter_defaults


def formulate_exponential_bugg_breit_wigner(
    decay_chain: ThreeBodyDecayChain,
) -> tuple[sp.Expr, dict[sp.Symbol, complex | float]]:
    """See `this paper, Eq. (4) <https://arxiv.org/pdf/hep-ex/0510019.pdf#page=3>`_."""
    expression, parameter_defaults = formulate_bugg_breit_wigner(decay_chain)
    alpha = create_alpha_symbol(decay_chain.resonance)
    parameter_defaults[alpha] = sp.Rational(0)
    s = get_mandelstam_s(decay_chain.decay_node)
    m0, m1 = sp.symbols("m0 m1", nonnegative=True)
    q = BreakupMomentum(m0**2, sp.sqrt(s), m1)
    expression *= sp.exp(-alpha * q**2)
    return expression, parameter_defaults


def formulate_flatte_1405(  # noqa: PLR0914
    decay_chain: ThreeBodyDecayChain,
) -> tuple[FlattéSWave, dict[sp.Symbol, complex | float]]:
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
    q = BreakupMomentum(m_top**2, sp.sqrt(s), m_spec)
    q0 = BreakupMomentum(m_top**2, m_res, m_spec)
    expression = sp.Mul(
        (q / q0) ** l_prod,
        BlattWeisskopf(q * R_prod, l_prod) / BlattWeisskopf(q0 * R_prod, l_prod),
        FlattéSWave(s, m_res, (Γ1, Γ2), (m1, m2), (mπ, mΣ)),
    )
    parameter_defaults: dict[sp.Symbol, complex | float] = {
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
    }
    return expression, parameter_defaults


def formulate_breit_wigner(
    decay_chain: ThreeBodyDecayChain,
) -> tuple[BreitWignerMinL, dict[sp.Symbol, complex | float]]:
    s = get_mandelstam_s(decay_chain.decay_node)
    m1, m2 = _create_decay_product_masses(decay_chain)
    l_dec = _get_angular_momentum(decay_chain.decay_node)
    l_prod = _get_angular_momentum(decay_chain.production_node)
    parent_mass = create_mass_symbol(decay_chain.parent)
    spectator_mass = create_mass_symbol(decay_chain.spectator)
    resonance_mass = create_mass_symbol(decay_chain.resonance)
    resonance_width = create_width_symbol(decay_chain.resonance)
    R_dec = create_meson_radius_symbol("dec")
    R_prod = create_meson_radius_symbol("prod")
    expression = BreitWignerMinL(
        s,
        parent_mass,
        spectator_mass,
        resonance_mass,
        resonance_width,
        m1,
        m2,
        l_dec,
        l_prod,
        R_dec,
        R_prod,
    )
    parameter_defaults: dict[sp.Symbol, complex | float] = {
        parent_mass: decay_chain.parent.mass,
        spectator_mass: decay_chain.spectator.mass,
        resonance_mass: decay_chain.resonance.mass,
        resonance_width: decay_chain.resonance.width,
        m1: decay_chain.decay_products[0].mass,
        m2: decay_chain.decay_products[1].mass,
        # https://github.com/ComPWA/polarimetry/pull/11#issuecomment-1128784376
        R_dec: 1.5,
        R_prod: 5,
    }
    return expression, parameter_defaults


def _create_decay_product_masses(
    decay_chain: ThreeBodyDecayChain,
) -> tuple[sp.Symbol, sp.Symbol]:
    mi, mj = (create_mass_symbol(p) for p in decay_chain.decay_products)
    return mi, mj

"""Dynamics lineshape definitions for the LHCb amplitude model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp
from ampform_dpd.dynamics import (
    BlattWeisskopf,
    BreitWignerMinL,
    BuggBreitWigner,
    FlattéSWave,
    Q,
)

from polarimetry.lhcb.particle import PARTICLE_TO_ID, Σ, K, p, π

if TYPE_CHECKING:
    from ampform_dpd.decay import Particle, ThreeBodyDecayChain


def formulate_bugg_breit_wigner(
    decay_chain: ThreeBodyDecayChain,
) -> tuple[BuggBreitWigner, dict[sp.Symbol, float | complex]]:
    if set(decay_chain.decay_products) != {π, K}:
        msg = "Bugg Breit-Wigner only defined for K* → Kπ"
        raise ValueError(msg)
    s = _get_mandelstam_s(decay_chain)
    m2, m3 = sp.symbols("m2 m3", nonnegative=True)
    gamma = sp.Symbol(Rf"\gamma_{{{decay_chain.resonance.name}}}")
    mass = sp.Symbol(f"m_{{{decay_chain.resonance.name}}}")
    width = sp.Symbol(Rf"\Gamma_{{{decay_chain.resonance.name}}}")
    parameter_defaults: dict[sp.Symbol, float | complex] = {
        mass: decay_chain.resonance.mass,
        width: decay_chain.resonance.width,
        m2: π.mass,
        m3: K.mass,
        gamma: 1,
    }
    expr = BuggBreitWigner(s, mass, width, m3, m2, gamma)  # Adler zero for K minus π
    return expr, parameter_defaults


def formulate_exponential_bugg_breit_wigner(
    decay_chain: ThreeBodyDecayChain,
) -> tuple[sp.Expr, dict[sp.Symbol, float | complex]]:
    """See `this paper, Eq. (4) <https://arxiv.org/pdf/hep-ex/0510019.pdf#page=3>`_."""
    expr, parameter_defaults = formulate_bugg_breit_wigner(decay_chain)
    alpha = sp.Symbol(Rf"\alpha_{{{decay_chain.resonance.name}}}")
    parameter_defaults[alpha] = sp.Rational(0)
    s = _get_mandelstam_s(decay_chain)
    m0, m1 = sp.symbols("m0 m1", nonnegative=True)
    q = Q(s, m0, m1)
    expr *= sp.exp(-alpha * q**2)
    return expr, parameter_defaults


def formulate_flatte_1405(  # noqa: PLR0914
    decay_chain: ThreeBodyDecayChain,
) -> tuple[FlattéSWave, dict[sp.Symbol, float | complex]]:
    if decay_chain.incoming_ls is None:
        msg = "Incoming L and S couplings are required for L(1405)"
        raise ValueError(msg)
    s = _get_mandelstam_s(decay_chain)
    m1, m2 = map(_to_mass_symbol, decay_chain.decay_products)
    m_res = sp.Symbol(f"m_{{{decay_chain.resonance.name}}}")
    Γ1 = sp.Symbol(Rf"\Gamma_{{{decay_chain.resonance.name} \to {p.latex} {K.latex}}}")
    Γ2 = sp.Symbol(Rf"\Gamma_{{{decay_chain.resonance.name} \to {Σ.latex} {π.latex}}}")
    m_top = _to_mass_symbol(decay_chain.parent)
    m_spec = _to_mass_symbol(decay_chain.spectator)
    mπ = _to_mass_symbol(π)
    mΣ = sp.Symbol(f"m_{{{Σ.name}}}")
    parameter_defaults: dict[sp.Symbol, float | complex] = {
        m_res: decay_chain.resonance.mass,
        Γ1: decay_chain.resonance.width,
        Γ2: decay_chain.resonance.width,
        m1: decay_chain.decay_products[0].mass,
        m2: decay_chain.decay_products[1].mass,
        m_top: decay_chain.parent.mass,
        m_spec: decay_chain.spectator.mass,
        mπ: π.mass,
        mΣ: Σ.mass,
    }
    l_prod = decay_chain.incoming_ls.L
    R_prod = sp.Symbol(R"R_{\Lambda_c}")
    q = Q(s, m_top, m_spec)
    q0 = Q(m_res**2, m_top, m_spec)
    dynamics = sp.Mul(
        (q / q0) ** l_prod,
        BlattWeisskopf(q * R_prod, l_prod) / BlattWeisskopf(q0 * R_prod, l_prod),
        FlattéSWave(s, m_res, (Γ1, Γ2), (m1, m2), (mπ, mΣ)),
    )
    return dynamics, parameter_defaults


def formulate_breit_wigner(
    decay_chain: ThreeBodyDecayChain,
) -> tuple[BreitWignerMinL, dict[sp.Symbol, float | complex]]:
    if decay_chain.incoming_ls is None:
        msg = "Incoming L and S couplings are required for Breit-Wigner"
        raise ValueError(msg)
    if decay_chain.outgoing_ls is None:
        msg = "Outgoing L and S couplings are required for Breit-Wigner"
        raise ValueError(msg)
    s = _get_mandelstam_s(decay_chain)
    child1_mass, child2_mass = map(_to_mass_symbol, decay_chain.decay_products)
    l_dec = sp.Rational(decay_chain.outgoing_ls.L)
    l_prod = sp.Rational(decay_chain.incoming_ls.L)
    parent_mass = sp.Symbol(f"m_{{{decay_chain.parent.name}}}")
    spectator_mass = sp.Symbol(f"m_{{{decay_chain.spectator.name}}}")
    resonance_mass = sp.Symbol(f"m_{{{decay_chain.resonance.name}}}")
    resonance_width = sp.Symbol(Rf"\Gamma_{{{decay_chain.resonance.name}}}")
    R_dec = sp.Symbol(R"R_\mathrm{res}")
    R_prod = sp.Symbol(R"R_{\Lambda_c}")
    parameter_defaults: dict[sp.Symbol, float | complex] = {
        parent_mass: decay_chain.parent.mass,
        spectator_mass: decay_chain.spectator.mass,
        resonance_mass: decay_chain.resonance.mass,
        resonance_width: decay_chain.resonance.width,
        child1_mass: decay_chain.decay_products[0].mass,
        child2_mass: decay_chain.decay_products[1].mass,
        # https://github.com/ComPWA/polarimetry/pull/11#issuecomment-1128784376
        R_dec: 1.5,
        R_prod: 5,
    }
    dynamics = BreitWignerMinL(
        s,
        parent_mass,
        spectator_mass,
        resonance_mass,
        resonance_width,
        child1_mass,
        child2_mass,
        l_dec,
        l_prod,
        R_dec,
        R_prod,
    )
    return dynamics, parameter_defaults


def _get_mandelstam_s(decay: ThreeBodyDecayChain) -> sp.Symbol:
    s1, s2, s3 = sp.symbols("sigma1:4", nonnegative=True)
    m1, m2, m3 = map(_to_mass_symbol, [p, π, K])
    decay_masses = {_to_mass_symbol(p) for p in decay.decay_products}
    if decay_masses == {m2, m3}:
        return s1
    if decay_masses == {m1, m3}:
        return s2
    if decay_masses == {m1, m2}:
        return s3
    decay_masses_str = "".join(str(s) for s in sorted(decay_masses, key=str))
    msg = f"Cannot find Mandelstam variable for {''.join(decay_masses_str)}"
    raise NotImplementedError(msg)


def _to_mass_symbol(particle: Particle) -> sp.Symbol:
    state_id = PARTICLE_TO_ID.get(particle)
    if state_id is not None:
        return sp.Symbol(f"m{state_id}", nonnegative=True)
    return sp.Symbol(f"m_{{{particle.name}}}", nonnegative=True)

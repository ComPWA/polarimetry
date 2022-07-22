from __future__ import annotations

import sympy as sp

from polarization.decay import Particle, ThreeBodyDecayChain
from polarization.dynamics import BreitWignerMinL, BuggBreitWigner, FlattéSWave, P

from .particle import PARTICLE_TO_ID, K, Σ, p, π


def formulate_bugg_breit_wigner(
    decay_chain: ThreeBodyDecayChain,
) -> tuple[BuggBreitWigner, dict[sp.Symbol, float]]:
    if set(decay_chain.decay_products) != {π, K}:
        raise ValueError("Bugg Breit-Wigner only defined for K* → Kπ")
    s = _get_mandelstam_s(decay_chain)
    m2, m3 = sp.symbols("m2 m3", nonnegative=True)
    gamma = sp.Symbol(Rf"\gamma_{{{decay_chain.resonance.name}}}")
    mass = sp.Symbol(f"m_{{{decay_chain.resonance.name}}}")
    width = sp.Symbol(Rf"\Gamma_{{{decay_chain.resonance.name}}}")
    parameter_defaults = {
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
) -> tuple[BuggBreitWigner, dict[sp.Symbol, float]]:
    expr, parameter_defaults = formulate_bugg_breit_wigner(decay_chain)
    alpha = sp.Symbol(Rf"\alpha_{{{decay_chain.resonance.name}}}")
    parameter_defaults[alpha] = sp.Rational(0)
    s = _get_mandelstam_s(decay_chain)
    m2, m3 = sp.symbols("m2 m3", nonnegative=True)
    q = P(s, m2, m3)
    expr *= sp.exp(-alpha * q**2)
    return expr, parameter_defaults


def formulate_flatte_1405(
    decay: ThreeBodyDecayChain,
) -> tuple[BuggBreitWigner, dict[sp.Symbol, float]]:
    s = _get_mandelstam_s(decay)
    m1, m2 = map(_to_mass_symbol, decay.decay_products)
    mass = sp.Symbol(f"m_{{{decay.resonance.name}}}")
    Γ1 = sp.Symbol(Rf"\Gamma_{{{decay.resonance.name} \to {p.latex} {K.latex}}}")
    Γ2 = sp.Symbol(Rf"\Gamma_{{{decay.resonance.name} \to {Σ.latex} {π.latex}}}")
    mπ = _to_mass_symbol(π)
    mΣ = sp.Symbol(f"m_{{{Σ.name}}}")
    parameter_defaults = {
        mass: decay.resonance.mass,
        Γ1: decay.resonance.width,
        Γ2: decay.resonance.width,
        m1: decay.decay_products[0].mass,
        m2: decay.decay_products[1].mass,
        mπ: π.mass,
        mΣ: Σ.mass,
    }
    dynamics = FlattéSWave(s, mass, (Γ1, Γ2), (m1, m2), (mπ, mΣ))
    return dynamics, parameter_defaults


def formulate_breit_wigner(decay_chain: ThreeBodyDecayChain):
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
    parameter_defaults = {
        parent_mass: decay_chain.parent.mass,
        spectator_mass: decay_chain.spectator.mass,
        resonance_mass: decay_chain.resonance.mass,
        resonance_width: decay_chain.resonance.width,
        child1_mass: decay_chain.decay_products[0].mass,
        child2_mass: decay_chain.decay_products[1].mass,
        # https://github.com/redeboer/polarization-sensitivity/pull/11#issuecomment-1128784376
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
    σ1, σ2, σ3 = sp.symbols("sigma1:4", nonnegative=True)
    m1, m2, m3 = map(_to_mass_symbol, [p, π, K])
    decay_masses = {_to_mass_symbol(p) for p in decay.decay_products}
    if decay_masses == {m2, m3}:
        return σ1
    if decay_masses == {m1, m3}:
        return σ2
    if decay_masses == {m1, m2}:
        return σ3
    raise NotImplementedError(
        f"Cannot find Mandelstam variable for {''.join(decay_masses)}"
    )


def _to_mass_symbol(particle: Particle) -> sp.Symbol:
    state_id = PARTICLE_TO_ID.get(particle)
    if state_id is not None:
        return sp.Symbol(f"m{state_id}", nonnegative=True)
    return sp.Symbol(f"m_{{{particle.name}}}", nonnegative=True)

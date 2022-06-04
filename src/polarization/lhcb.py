# cspell:ignore modelparameters modelstudies
# pyright: reportConstantRedefinition=false
"""Import functions that are specifically for this LHCb analysis.

.. seealso:: :doc:`/cross-check`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import sympy as sp
from sympy.core.symbol import Str

from polarization.amplitude import DynamicsBuilder, DynamicsConfigurator
from polarization.decay import IsobarNode, Particle, ThreeBodyDecay, ThreeBodyDecayChain
from polarization.dynamics import BreitWignerMinL, BuggBreitWigner, FlattéSWave
from polarization.spin import filter_parity_violating_ls, generate_ls_couplings

if sys.version_info < (3, 8):
    from typing_extensions import Literal, TypedDict
else:
    from typing import Literal, TypedDict

Λc = Particle(
    name="Λc⁺",
    latex=R"\Lambda_c^+",
    spin=0.5,
    parity=+1,
    mass=2.28646,
    width=3.25e-12,
)
p = Particle(
    name="p",
    latex="p",
    spin=0.5,
    parity=+1,
    mass=0.938272046,
    width=0.0,
)
K = Particle(
    name="K⁻",
    latex="K^-",
    spin=0,
    parity=-1,
    mass=0.493677,
    width=5.317e-17,
)
π = Particle(
    name="π⁺",
    latex=R"\pi^+",
    spin=0,
    parity=-1,
    mass=0.13957018,
    width=2.5284e-17,
)
_PARTICLE_TO_ID = {Λc: 0, p: 1, π: 2, K: 3}

# https://github.com/redeboer/polarization-sensitivity/blob/34f5330/julia/notebooks/model0.jl#L43-L47
Σ = Particle(
    name="Σ⁻",
    latex=R"\Sigma^-",
    spin=0.5,
    parity=+1,
    mass=1.18937,
    width=4.45e-15,
)


def load_three_body_decays(filename: str) -> DynamicsConfigurator:
    def create_isobar(resonance: Particle) -> ThreeBodyDecayChain:
        if resonance.name.startswith("K"):
            child1, child2, sibling = π, K, p
        elif resonance.name.startswith("L"):
            child1, child2, sibling = K, p, π
        elif resonance.name.startswith("D"):
            child1, child2, sibling = p, π, K
        else:
            raise NotImplementedError
        decay = IsobarNode(
            parent=Λc,
            child1=sibling,
            child2=IsobarNode(
                parent=resonance,
                child1=child1,
                child2=child2,
                interaction=generate_L_min(
                    resonance, child1, child2, conserve_parity=True
                ),
            ),
            interaction=generate_L_min(Λc, sibling, resonance, conserve_parity=False),
        )
        return ThreeBodyDecayChain(decay)

    def generate_L_min(
        parent: Particle, child1: Particle, child2: Particle, conserve_parity: bool
    ) -> int:
        ls = generate_ls_couplings(parent.spin, child1.spin, child2.spin)
        if conserve_parity:
            ls = filter_parity_violating_ls(
                ls, parent.parity, child1.parity, child2.parity
            )
        return min(ls)

    resonances = load_resonance_definitions(filename)
    decay = ThreeBodyDecay(
        states={state_id: particle for particle, state_id in _PARTICLE_TO_ID.items()},
        chains=tuple(create_isobar(res) for res in resonances.values()),
    )
    dynamics_configurator = DynamicsConfigurator(decay)
    for chain in decay.chains:
        builder = _get_resonance_builder(chain.resonance.lineshape)
        dynamics_configurator.register_builder(chain, builder)
    return dynamics_configurator


def _get_resonance_builder(lineshape: str) -> DynamicsBuilder:
    if lineshape == "BreitWignerMinL":
        return formulate_breit_wigner
    if lineshape == "BuggBreitWignerMinL":
        return formulate_bugg_breit_wigner
    if lineshape == "Flatte1405":
        return formulate_flatte_1405
    raise NotImplementedError(f'No dynamics implemented for lineshape "{lineshape}"')


def load_resonance_definitions(filename: Path | str) -> dict[str, Particle]:
    """Load `Particle` definitions from a JSON file."""
    with open(filename) as stream:
        data = json.load(stream)
    isobar_definitions = data["isobars"]
    return to_resonance_dict(isobar_definitions)


def load_model_parameters(
    filename: Path | str, decay: ThreeBodyDecay, model_number: int = 0
) -> dict[sp.Indexed | sp.Symbol, complex | float]:
    with open("../data/modelparameters.json") as stream:
        json_data = json.load(stream)
    json_parameters = json_data["modelstudies"][model_number]["parameters"]
    json_parameters["ArK(892)1"] = "1.0 ± 0.0"
    json_parameters["AiK(892)1"] = "0.0 ± 0.0"
    parameters = to_symbol_definitions(json_parameters, decay)
    decay_couplings = compute_decay_couplings(decay)
    parameters.update(decay_couplings)
    return parameters


def compute_decay_couplings(decay: ThreeBodyDecay) -> dict[sp.Indexed, Literal[-1, 1]]:
    H_dec = sp.IndexedBase(R"\mathcal{H}^\mathrm{decay}")
    half = sp.Rational(1, 2)
    decay_couplings = {}
    for chain in decay.chains:
        R = Str(chain.resonance.latex)
        if chain.resonance.name.startswith("K"):
            decay_couplings[H_dec[R, 0, 0]] = 1
        if chain.resonance.name[0] in {"D", "L"}:
            child1, child2 = chain.decay_products
            if chain.resonance.name.startswith("D"):
                coupling_pos = H_dec[R, +half, 0]
                coupling_neg = H_dec[R, -half, 0]
            else:
                coupling_pos = H_dec[R, 0, +half]
                coupling_neg = H_dec[R, 0, -half]
            decay_couplings[coupling_pos] = 1
            decay_couplings[coupling_neg] = int(
                chain.resonance.parity
                * child1.parity
                * child2.parity
                * (-1) ** (chain.resonance.spin - child1.spin - child2.spin)
            )
    return decay_couplings


def to_resonance_dict(definition: dict[str, ResonanceJSON]) -> dict[str, Particle]:
    return {
        name: to_resonance(name, resonance_def)
        for name, resonance_def in definition.items()
    }


def to_resonance(name: str, definition: ResonanceJSON) -> Particle:
    spin, parity = _to_jp_pair(definition["jp"])
    return Particle(
        name,
        name,
        spin,
        parity,
        mass=_average_float(definition["mass"]) * 1e-3,  # MeV to GeV
        width=_average_float(definition["width"]) * 1e-3,  # MeV to GeV
        lineshape=definition["lineshape"],
    )


def _average_float(input_str: str) -> tuple[float, float]:
    """
    >>> _average_float("1405.1")
    1405.1
    >>> _average_float("1900-2100")
    2000.0
    """
    if "-" in input_str:
        _min, _max, *_ = map(float, input_str.split("-"))
        return (_max + _min) / 2
    return float(input_str)


def _to_jp_pair(input_str: str) -> tuple[sp.Rational, int]:
    """
    >>> _to_jp_pair("3/2^-")
    (3/2, -1)
    >>> _to_jp_pair("0^+")
    (0, 1)
    """
    spin, parity_sign = input_str.split("^")
    return sp.Rational(spin), int(f"{parity_sign}1")


class ResonanceJSON(TypedDict):
    jp: str
    mass: str
    width: str
    lineshape: Literal["BreitWignerMinL", "BuggBreitWignerMinL", "Flatte1405"]


def to_symbol_definitions(
    parameter_dict: dict[str, str], decay: ThreeBodyDecay
) -> dict[sp.Basic, complex | float]:
    key_to_val: dict[str, complex | float] = {}
    for key, str_value in parameter_dict.items():
        if key.startswith("Ar"):
            identifier = key[2:]
            str_imag = parameter_dict[f"Ai{identifier}"]
            real = to_float(str_value)
            imag = to_float(str_imag)
            key = f"A{identifier}"
            indexed_symbol: sp.Indexed = to_symbol(key)
            chain = decay.find_chain(resonance_name=str(indexed_symbol.indices[0]))
            factor = get_conversion_factor(chain.resonance)
            key_to_val[f"A{identifier}"] = factor * complex(real, imag)
        elif key.startswith("Ai"):
            continue
        else:
            key_to_val[key] = to_float(str_value)
    return {to_symbol(key): value for key, value in key_to_val.items()}


def get_conversion_factor(
    resonance: Particle, proton_helicity: sp.Rational | None = None
) -> Literal[-1, 1]:
    half = sp.Rational(1, 2)
    factor = 1
    if proton_helicity is not None:
        factor = int((-1) ** (half - proton_helicity))  # two-particle convention
    if resonance.name.startswith("D"):
        return int(-resonance.parity * factor * (-1) ** (resonance.spin - half))
    if resonance.name.startswith("K"):
        return factor
    if resonance.name.startswith("L"):
        return int(-resonance.parity * factor)
    raise NotImplementedError(f"No conversion factor implemented for {resonance.name}")


def to_float(str_value: str) -> float:
    value, _ = map(float, str_value.split(" ± "))
    return value


def to_symbol(key: str) -> sp.Indexed | sp.Symbol:
    H_prod = sp.IndexedBase(R"\mathcal{H}^\mathrm{production}")
    half = sp.Rational(1, 2)
    if key.startswith("A"):
        # https://github.com/redeboer/polarization-sensitivity/issues/5#issue-1220525993
        R = stringify(key[1:-1])
        i = int(key[-1])
        if str(R).startswith("L"):
            if i == 1:
                return H_prod[R, -half, 0]
            if i == 2:
                return H_prod[R, +half, 0]
        if str(R).startswith("D"):
            if i == 1:
                return H_prod[R, -half, 0]
            if i == 2:
                return H_prod[R, +half, 0]
        if str(R).startswith("K"):
            if str(R) in {"K(700)", "K(1430)"}:
                if i == 1:
                    return H_prod[R, 0, +half]
                if i == 2:
                    return H_prod[R, 0, -half]
            else:
                if i == 1:
                    return H_prod[R, 0, -half]
                if i == 2:
                    return H_prod[R, -1, -half]
                if i == 3:
                    return H_prod[R, +1, +half]
                if i == 4:
                    return H_prod[R, 0, +half]
    if key.startswith("gamma"):
        R = stringify(key[5:])
        return sp.Symbol(Rf"\gamma_{{{R}}}")
    if key.startswith("M"):
        R = stringify(key[1:])
        return sp.Symbol(Rf"m_{{{R}}}")
    if key.startswith("G"):
        R = stringify(key[1:])
        return sp.Symbol(Rf"\Gamma_{{{R}}}")
    raise NotImplementedError(
        f'Cannot convert key "{key}" in model parameter JSON file to SymPy symbol'
    )


def stringify(obj) -> Str:
    if isinstance(obj, Particle):
        return Str(obj.latex)
    return Str(f"{obj}")


def formulate_breit_wigner(decay_chain: ThreeBodyDecayChain):
    s = get_mandelstam_s(decay_chain)
    child1_mass, child2_mass = map(to_mass_symbol, decay_chain.decay_products)
    l_dec = sp.Rational(decay_chain.outgoing_ls.L)
    l_prod = sp.Rational(decay_chain.incoming_ls.L)
    parent_mass = sp.Symbol(f"m_{{{decay_chain.parent.latex}}}")
    spectator_mass = sp.Symbol(f"m_{{{decay_chain.spectator.latex}}}")
    resonance_mass = sp.Symbol(f"m_{{{decay_chain.resonance.latex}}}")
    resonance_width = sp.Symbol(Rf"\Gamma_{{{decay_chain.resonance.latex}}}")
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


def formulate_bugg_breit_wigner(
    decay_chain: ThreeBodyDecayChain,
) -> tuple[BuggBreitWigner, dict[sp.Symbol, float]]:
    if set(decay_chain.decay_products) != {π, K}:
        raise ValueError("Bugg Breit-Wigner only defined for K* → Kπ")
    s = get_mandelstam_s(decay_chain)
    m2, m3 = sp.symbols("m2 m3", nonnegative=True)
    gamma = sp.Symbol(Rf"\gamma_{{{decay_chain.resonance.latex}}}")
    mass = sp.Symbol(f"m_{{{decay_chain.resonance.latex}}}")
    width = sp.Symbol(Rf"\Gamma_{{{decay_chain.resonance.latex}}}")
    parameter_defaults = {
        mass: decay_chain.resonance.mass,
        width: decay_chain.resonance.width,
        m2: π.mass,
        m3: K.mass,
        gamma: 1,
    }
    expr = BuggBreitWigner(s, mass, width, m3, m2, gamma)  # Adler zero for K minus π
    return expr, parameter_defaults


def formulate_flatte_1405(
    decay: ThreeBodyDecayChain,
) -> tuple[BuggBreitWigner, dict[sp.Symbol, float]]:
    s = get_mandelstam_s(decay)
    m1, m2 = map(to_mass_symbol, decay.decay_products)
    mass = sp.Symbol(f"m_{{{decay.resonance.latex}}}")
    width = sp.Symbol(Rf"\Gamma_{{{decay.resonance.latex}}}")
    mπ = to_mass_symbol(π)
    mΣ = sp.Symbol(f"m_{{{Σ.latex}}}")
    parameter_defaults = {
        mass: decay.resonance.mass,
        width: decay.resonance.width,
        m1: decay.decay_products[0].mass,
        m2: decay.decay_products[1].mass,
        mπ: π.mass,
        mΣ: Σ.mass,
    }
    dynamics = FlattéSWave(s, mass, width, (m1, m2), (mπ, mΣ))
    return dynamics, parameter_defaults


def get_mandelstam_s(decay: ThreeBodyDecayChain) -> sp.Symbol:
    σ1, σ2, σ3 = sp.symbols("sigma1:4", nonnegative=True)
    m1, m2, m3 = map(to_mass_symbol, [p, π, K])
    decay_masses = {to_mass_symbol(p) for p in decay.decay_products}
    if decay_masses == {m2, m3}:
        return σ1
    if decay_masses == {m1, m3}:
        return σ2
    if decay_masses == {m1, m2}:
        return σ3
    raise NotImplementedError(
        f"Cannot find Mandelstam variable for {''.join(decay_masses)}"
    )


def to_mass_symbol(particle: Particle) -> sp.Symbol:
    state_id = _PARTICLE_TO_ID.get(particle)
    if state_id is not None:
        return sp.Symbol(f"m{state_id}", nonnegative=True)
    return sp.Symbol(f"m_{{{particle.latex}}}", nonnegative=True)

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
from polarization.spin import filter_parity_violating_ls, generate_ls_couplings

from .dynamics import (
    formulate_breit_wigner,
    formulate_bugg_breit_wigner,
    formulate_flatte_1405,
)
from .particle import PARTICLE_TO_ID, K, Λc, p, π

if sys.version_info < (3, 8):
    from typing_extensions import Literal, TypedDict
else:
    from typing import Literal, TypedDict


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
        states={state_id: particle for particle, state_id in PARTICLE_TO_ID.items()},
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
    return _to_resonance_dict(isobar_definitions)


def load_model_parameters(
    filename: Path | str,
    decay: ThreeBodyDecay,
    model_id: int | str = 0,
    typ: Literal["value", "uncertainty"] = "value",
) -> dict[sp.Indexed | sp.Symbol, complex | float]:
    with open(filename) as stream:
        json_data = json.load(stream)
    if isinstance(model_id, str):
        model_id = _get_model_by_title(json_data, model_id)
    json_parameters = json_data["modelstudies"][model_id]["parameters"]
    json_parameters["ArK(892)1"] = "1.0 ± 0.0"
    json_parameters["AiK(892)1"] = "0.0 ± 0.0"
    parameters = _to_symbol_value_mapping(json_parameters, decay, typ)
    decay_couplings = compute_decay_couplings(decay, typ)
    parameters.update(decay_couplings)
    return parameters


def _get_model_by_title(json_data: dict, title: str) -> int:
    for i, item in enumerate(json_data["modelstudies"]):
        title = item["title"]
        if item["title"] == title:
            return i
    raise KeyError(f'Could not find model with title "{title}"')


def compute_decay_couplings(
    decay: ThreeBodyDecay, typ: Literal["value", "uncertainty"] = "value"
) -> dict[sp.Indexed, Literal[-1, 0, 1]]:
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
    if typ == "uncertainty":
        return {s: 0 for s in decay_couplings}
    return decay_couplings


def _to_resonance_dict(definition: dict[str, ResonanceJSON]) -> dict[str, Particle]:
    return {
        name: _to_resonance(name, resonance_def)
        for name, resonance_def in definition.items()
    }


def _to_resonance(name: str, definition: ResonanceJSON) -> Particle:
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


def _to_jp_pair(input_str: str) -> tuple[sp.Rational, int]:
    """
    >>> _to_jp_pair("3/2^-")
    (3/2, -1)
    >>> _to_jp_pair("0^+")
    (0, 1)
    """
    spin, parity_sign = input_str.split("^")
    return sp.Rational(spin), int(f"{parity_sign}1")


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


class ResonanceJSON(TypedDict):
    jp: str
    mass: str
    width: str
    lineshape: Literal["BreitWignerMinL", "BuggBreitWignerMinL", "Flatte1405"]


def _to_symbol_value_mapping(
    parameter_dict: dict[str, str],
    decay: ThreeBodyDecay,
    typ: Literal["value", "uncertainty"] = "value",
) -> dict[sp.Basic, complex | float]:
    switch = 0 if typ == "value" else 1
    key_to_val: dict[str, complex | float] = {}
    for key, str_value in parameter_dict.items():
        if key.startswith("Ar"):
            identifier = key[2:]
            str_imag = parameter_dict[f"Ai{identifier}"]
            real = _to_value_with_uncertainty(str_value)[switch]
            imag = _to_value_with_uncertainty(str_imag)[switch]
            key = f"A{identifier}"
            indexed_symbol: sp.Indexed = parameter_key_to_symbol(key)
            chain = decay.find_chain(resonance_name=str(indexed_symbol.indices[0]))
            if typ == "value":
                factor = get_conversion_factor(chain.resonance)
            else:
                factor = 1
            key_to_val[f"A{identifier}"] = factor * complex(real, imag)
        elif key.startswith("Ai"):
            continue
        else:
            key_to_val[key] = _to_value_with_uncertainty(str_value)[switch]
    return {parameter_key_to_symbol(key): value for key, value in key_to_val.items()}


def _to_value_with_uncertainty(str_value: str) -> float:
    """
    >>> _to_value_with_uncertainty('1.5 ± 0.2')
    (1.5, 0.2)
    """
    value, uncertainty = map(float, str_value.split(" ± "))
    return value, uncertainty


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


def parameter_key_to_symbol(key: str) -> sp.Indexed | sp.Symbol:
    H_prod = sp.IndexedBase(R"\mathcal{H}^\mathrm{production}")
    half = sp.Rational(1, 2)
    if key.startswith("A"):
        # https://github.com/redeboer/polarization-sensitivity/issues/5#issue-1220525993
        R = _stringify(key[1:-1])
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
        R = _stringify(key[5:])
        return sp.Symbol(Rf"\gamma_{{{R}}}")
    if key.startswith("M"):
        R = _stringify(key[1:])
        return sp.Symbol(Rf"m_{{{R}}}")
    if key.startswith("G"):
        R = _stringify(key[1:])
        return sp.Symbol(Rf"\Gamma_{{{R}}}")
    raise NotImplementedError(
        f'Cannot convert key "{key}" in model parameter JSON file to SymPy symbol'
    )


def _stringify(obj) -> Str:
    if isinstance(obj, Particle):
        return Str(obj.latex)
    return Str(f"{obj}")

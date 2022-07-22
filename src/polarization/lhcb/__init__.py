# cspell:ignore modelparameters modelstudies
# pyright: reportConstantRedefinition=false
"""Import functions that are specifically for this LHCb analysis.

.. seealso:: :doc:`/cross-check`
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

import attrs
import numpy as np
import sympy as sp
import yaml
from sympy.core.symbol import Str

from polarization.amplitude import (
    AmplitudeModel,
    DalitzPlotDecompositionBuilder,
    DynamicsBuilder,
)
from polarization.decay import IsobarNode, Particle, ThreeBodyDecay, ThreeBodyDecayChain
from polarization.spin import filter_parity_violating_ls, generate_ls_couplings

from .dynamics import (
    formulate_breit_wigner,
    formulate_bugg_breit_wigner,
    formulate_exponential_bugg_breit_wigner,
    formulate_flatte_1405,
)
from .particle import PARTICLE_TO_ID, K, Λc, Σ, p, π

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal


def load_model(
    model_file: Path | str,
    particle_definitions: dict[str, Particle],
    model_id: int | str = 0,
) -> AmplitudeModel:
    builder = load_model_builder(model_file, particle_definitions, model_id)
    model = builder.formulate()
    imported_parameter_values = load_model_parameters(
        model_file, builder.decay, model_id
    )
    model.parameter_defaults.update(imported_parameter_values)
    return model


def load_model_builder(
    model_file: Path | str,
    particle_definitions: dict[str, Particle],
    model_id: int | str = 0,
) -> DalitzPlotDecompositionBuilder:
    with open(model_file) as f:
        model_definitions = yaml.load(f, Loader=yaml.SafeLoader)
    model_title = _find_model_title(model_definitions, model_id)
    model_def = model_definitions[model_title]
    lineshapes: dict[str, str] = model_def["lineshapes"]
    decay = load_three_body_decay(lineshapes, particle_definitions)
    amplitude_builder = DalitzPlotDecompositionBuilder(decay)
    for chain in decay.chains:
        lineshape_choice = lineshapes[chain.resonance.name]
        dynamics_builder = _get_resonance_builder(lineshape_choice)
        amplitude_builder.dynamics_choices.register_builder(chain, dynamics_builder)
    return amplitude_builder


def _find_model_title(model_definitions: dict, model_id: int | str) -> str:
    if isinstance(model_id, int):
        if model_id >= len(model_definitions):
            raise KeyError(
                f"Model definition file contains {len(model_definitions)} models, but"
                f" trying to get number {model_id}."
            )
        for i, title in enumerate(model_definitions):
            if i == model_id:
                return title
    if model_id not in model_definitions:
        raise KeyError(f'Could not find model with title "{model_id}"')
    return model_id


def _get_resonance_builder(lineshape: str) -> DynamicsBuilder:
    if lineshape == "BreitWignerMinL":
        return formulate_breit_wigner
    if lineshape == "BuggBreitWignerExpFF":
        return formulate_exponential_bugg_breit_wigner
    if lineshape == "BuggBreitWignerMinL":
        return formulate_bugg_breit_wigner
    if lineshape == "Flatte1405":
        return formulate_flatte_1405
    raise NotImplementedError(f'No dynamics implemented for lineshape "{lineshape}"')


def load_three_body_decay(
    resonance_names: Iterable[str],
    particle_definitions: dict[str, Particle],
) -> ThreeBodyDecay:
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

    resonances = [particle_definitions[name] for name in resonance_names]
    return ThreeBodyDecay(
        states={state_id: particle for particle, state_id in PARTICLE_TO_ID.items()},
        chains=tuple(create_isobar(res) for res in resonances),
    )


class ParameterBootstrap:
    """A wrapper for loading parameters from :download:`model-definitions.yaml </../data/model-definitions.yaml>`."""

    def __init__(
        self,
        filename: Path | str,
        decay: ThreeBodyDecay,
        model_id: int | str = 0,
    ) -> None:
        self.__values = {
            str(symbol): value
            for symbol, value in load_model_parameters(
                filename, decay, model_id, typ="value"
            ).items()
        }
        self.__uncertainties = {
            str(symbol): value
            for symbol, value in load_model_parameters(
                filename, decay, model_id, typ="uncertainty"
            ).items()
        }

    @property
    def values(self) -> dict[str, complex | float | int]:
        return dict(self.__values)

    @property
    def uncertainties(self) -> dict[str, complex | float | int]:
        return dict(self.__uncertainties)

    def create_distribution(
        self, sample_size: int, seed: int | None = None
    ) -> dict[str, complex | float | int]:
        return _smear_gaussian(
            parameter_values=self.values,
            parameter_uncertainties=self.uncertainties,
            size=sample_size,
            seed=seed,
        )


def load_model_parameters(
    filename: Path | str,
    decay: ThreeBodyDecay,
    model_id: int | str = 0,
    typ: Literal["value", "uncertainty"] = "value",
) -> dict[sp.Indexed | sp.Symbol, complex | float]:
    with open(filename) as f:
        model_definitions = yaml.load(f, Loader=yaml.SafeLoader)
    model_title = _find_model_title(model_definitions, model_id)
    parameter_definitions = model_definitions[model_title]["parameters"]
    parameters = _to_symbol_value_mapping(parameter_definitions, decay, typ)
    decay_couplings = compute_decay_couplings(decay, typ)
    parameters.update(decay_couplings)
    return parameters


def _smear_gaussian(
    parameter_values: dict[str, complex | float],
    parameter_uncertainties: dict[str, complex | float],
    size: int,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    value_distributions = {}
    for k, mean in parameter_values.items():
        std = parameter_uncertainties[k]
        distribution = _create_gaussian_distribution(mean, std, size, seed)
        value_distributions[k] = distribution
    return value_distributions


def _create_gaussian_distribution(
    mean: complex | float,
    std: complex | float,
    size: int,
    seed: int | None = None,
):
    rng = np.random.default_rng(seed)
    if isinstance(mean, complex) and isinstance(std, complex):
        return (
            rng.normal(mean.real, std.real, size)
            + rng.normal(mean.imag, std.imag, size) * 1j
        )
    if isinstance(mean, (float, int)) and isinstance(std, (float, int)):
        return rng.normal(mean, std, size)
    raise NotImplementedError


def flip_production_coupling_signs(
    model: AmplitudeModel, subsystem_names: list[str]
) -> AmplitudeModel:
    new_parameters = {}
    name_pattern = rf".*\\mathrm{{production}}\[[{''.join(subsystem_names)}].*"
    for symbol, value in model.parameter_defaults.items():
        if re.match(name_pattern, str(symbol)) is not None:
            value *= -1
        new_parameters[symbol] = value
    return attrs.evolve(model, parameter_defaults=new_parameters)


def compute_decay_couplings(
    decay: ThreeBodyDecay, typ: Literal["value", "uncertainty"] = "value"
) -> dict[sp.Indexed, int]:
    H_dec = sp.IndexedBase(R"\mathcal{H}^\mathrm{decay}")
    half = sp.Rational(1, 2)
    decay_couplings = {}
    for chain in decay.chains:
        R = Str(chain.resonance.name)
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
    if key.startswith("alpha"):
        R = _stringify(key[5:])
        return sp.Symbol(Rf"\alpha_{{{R}}}")
    if key.startswith("gamma"):
        R = _stringify(key[5:])
        return sp.Symbol(Rf"\gamma_{{{R}}}")
    if key.startswith("M"):
        R = _stringify(key[1:])
        return sp.Symbol(Rf"m_{{{R}}}")
    if key.startswith("G1"):
        R = _stringify(key[2:])
        return sp.Symbol(Rf"\Gamma_{{{R} \to {p.latex} {K.latex}}}")
    if key.startswith("G2"):
        R = _stringify(key[2:])
        return sp.Symbol(Rf"\Gamma_{{{R} \to {Σ.latex} {π.latex}}}")
    if key.startswith("G"):
        R = _stringify(key[1:])
        return sp.Symbol(Rf"\Gamma_{{{R}}}")
    if key == "dLc":
        return sp.Symbol(R"R_{\Lambda_c}")
    raise NotImplementedError(
        f'Cannot convert key "{key}" in model parameter JSON file to SymPy symbol'
    )


def _stringify(obj) -> Str:
    if isinstance(obj, Particle):
        return Str(obj.name)
    return Str(f"{obj}")

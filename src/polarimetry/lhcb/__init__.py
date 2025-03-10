# cspell:ignore modelparameters modelstudies
# pyright: reportConstantRedefinition=false
"""Import functions that are specifically for this LHCb analysis.

.. seealso:: :doc:`/cross-check`
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable
from copy import deepcopy
from math import sqrt
from typing import TYPE_CHECKING, Generic, Literal, TypedDict, TypeVar

import attrs
import numpy as np
import sympy as sp
import yaml
from ampform_dpd import (
    AmplitudeModel,
    DalitzPlotDecompositionBuilder,
    DynamicsBuilder,
    _get_coupling_base,  # pyright:ignore[reportPrivateUsage]  # noqa: PLC2701
)
from ampform_dpd.decay import IsobarNode, Particle, ThreeBodyDecay, ThreeBodyDecayChain
from ampform_dpd.spin import filter_parity_violating_ls, generate_ls_couplings
from attrs import frozen
from sympy.core.symbol import Str

from polarimetry.lhcb.dynamics import (
    formulate_breit_wigner,
    formulate_bugg_breit_wigner,
    formulate_exponential_bugg_breit_wigner,
    formulate_flatte_1405,
)
from polarimetry.lhcb.particle import Σ, K, Λc, p, π

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from ampform_dpd.decay import State


# fmt: off
ModelName = Literal[
    "Default amplitude model",
    "Alternative amplitude model with K(892) with free mass and width",
    "Alternative amplitude model with L(1670) with free mass and width",
    "Alternative amplitude model with L(1690) with free mass and width",
    "Alternative amplitude model with D(1232) with free mass and width",
    "Alternative amplitude model with L(1600), D(1600), D(1700) with free mass and width",
    "Alternative amplitude model with free L(1405) Flatt'e widths, indicated as G1 (pK channel) and G2 (Sigmapi)",
    "Alternative amplitude model with L(1800) contribution added with free mass and width",
    "Alternative amplitude model with L(1810) contribution added with free mass and width",
    "Alternative amplitude model with D(1620) contribution added with free mass and width",
    "Alternative amplitude model in which a Relativistic Breit-Wigner is used for the K(700) contribution",
    "Alternative amplitude model with K(700) with free mass and width",
    "Alternative amplitude model with K(1410) contribution added with mass and width from PDG2020",
    "Alternative amplitude model in which a Relativistic Breit-Wigner is used for the K(1430) contribution",
    "Alternative amplitude model with K(1430) with free width",
    "Alternative amplitude model with an additional overall exponential form factor exp(-alpha q^2) multiplying Bugg lineshapes. The exponential parameter is indicated as alpha",
    "Alternative amplitude model with free radial parameter d for the Lc resonance, indicated as dLc",
    "Alternative amplitude model obtained using LS couplings",
]
"""The names of the published models."""
# fmt: on
LineshapeName = Literal[
    "BreitWignerMinL",
    "BreitWignerMinL_LS",
    "BuggBreitWignerExpFF",
    "BuggBreitWignerMinL",
    "BuggBreitWignerMinL_LS",
    "Flatte1405",
    "Flatte1405_LS",
]
"""Allowed lineshape names in the model definition file."""


ResonanceName = Literal[
    "D(1232)",
    "D(1600)",
    "D(1620)",
    "D(1700)",
    "K(1410)",
    "K(1430)",
    "K(700)",
    "K(892)",
    "L(1405)",
    "L(1520)",
    "L(1600)",
    "L(1670)",
    "L(1690)",
    "L(1800)",
    "L(1810)",
    "L(2000)",
]
"""Allowed resonance names in the model definition file."""


class ModelDefinition(TypedDict):
    parameters: dict[str, str]
    lineshapes: dict[ResonanceName, LineshapeName]


def load_model(
    model_file: Path | str,
    particle_definitions: dict[str, Particle],
    model_id: int | ModelName = 0,
    cleanup_summations: bool = False,
) -> AmplitudeModel:
    builder = load_model_builder(model_file, particle_definitions, model_id)
    model = builder.formulate(cleanup_summations=cleanup_summations)
    imported_parameter_values = load_model_parameters(
        model_file, builder.decay, model_id, particle_definitions
    )
    model.parameter_defaults.update(imported_parameter_values)  # type:ignore[arg-type]  # pyright:ignore[reportCallIssue]
    return model


def load_model_builder(
    model_file: Path | str,
    particle_definitions: dict[str, Particle],
    model_id: int | ModelName = 0,
) -> DalitzPlotDecompositionBuilder:
    model_definitions = _load_model_definitions(model_file)
    model_title = _find_model_title(model_definitions, model_id)
    model_def = model_definitions[model_title]
    lineshapes = model_def["lineshapes"]
    min_ls = "LS couplings" not in model_title
    decay = load_three_body_decay(lineshapes, particle_definitions, min_ls)
    amplitude_builder = DalitzPlotDecompositionBuilder(decay, min_ls=(min_ls, True))
    for chain in decay.chains:
        lineshape_choice = lineshapes[chain.resonance.name]  # type:ignore[index]
        dynamics_builder = _get_resonance_builder(lineshape_choice)
        amplitude_builder.dynamics_choices.register_builder(chain, dynamics_builder)
    return amplitude_builder


def _load_model_definitions(model_file: Path | str) -> dict[ModelName, ModelDefinition]:
    with open(model_file) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def _find_model_title(
    model_definitions: dict[ModelName, ModelDefinition],
    model_id: int | ModelName,
) -> ModelName:
    if isinstance(model_id, int):
        for i, title in enumerate(model_definitions):
            if i == model_id:
                return title
        msg = (
            f"Model definition file contains {len(model_definitions)} models, but"
            f" trying to get number {model_id}."
        )
        raise KeyError(msg)
    if model_id not in model_definitions:
        msg = f'Could not find model with title "{model_id}"'
        raise KeyError(msg)
    return model_id


def _get_resonance_builder(lineshape: LineshapeName) -> DynamicsBuilder:
    if lineshape in {"BreitWignerMinL", "BreitWignerMinL_LS"}:
        return formulate_breit_wigner
    if lineshape == "BuggBreitWignerExpFF":
        return formulate_exponential_bugg_breit_wigner
    if lineshape in {"BuggBreitWignerMinL", "BuggBreitWignerMinL_LS"}:
        return formulate_bugg_breit_wigner
    if lineshape in {"Flatte1405", "Flatte1405_LS"}:
        return formulate_flatte_1405
    msg = f'No dynamics implemented for lineshape "{lineshape}"'
    raise NotImplementedError(msg)


def load_three_body_decay(
    resonance_names: Iterable[ResonanceName],
    particle_definitions: dict[str, Particle],
    min_ls: bool = True,
) -> ThreeBodyDecay:
    def create_isobar(resonance: Particle) -> list[ThreeBodyDecayChain]:
        if resonance.name.startswith("K"):
            child1, child2, spectator = π, K, p
        elif resonance.name.startswith("L"):
            child1, child2, spectator = K, p, π
        elif resonance.name.startswith("D"):
            child1, child2, spectator = p, π, K
        else:
            msg = f"Unknown which decay products to assign to {resonance.name}"
            raise NotImplementedError(msg)
        prod_ls_couplings = generate_ls(Λc, resonance, spectator, conserve_parity=False)
        dec_ls_couplings = generate_ls(resonance, child1, child2, conserve_parity=True)
        if min_ls:
            decay = IsobarNode(
                parent=Λc,
                child1=IsobarNode(
                    parent=resonance,
                    child1=child1,
                    child2=child2,
                    interaction=min(dec_ls_couplings),
                ),
                child2=spectator,
                interaction=min(prod_ls_couplings),
            )
            return [ThreeBodyDecayChain(decay)]
        chains = []
        for dec_ls, prod_ls in itertools.product(dec_ls_couplings, prod_ls_couplings):
            decay = IsobarNode(
                parent=Λc,
                child1=IsobarNode(
                    parent=resonance,
                    child1=child1,
                    child2=child2,
                    interaction=dec_ls,
                ),
                child2=spectator,
                interaction=prod_ls,
            )
            chains.append(ThreeBodyDecayChain(decay))
        return chains

    def generate_ls(
        parent: Particle, child1: Particle, child2: Particle, conserve_parity: bool
    ) -> list[tuple[int, sp.Rational]]:
        ls = generate_ls_couplings(parent.spin, child1.spin, child2.spin)
        if conserve_parity:
            msg = "Parity is not defined"
            assert parent.parity is not None, msg  # noqa: S101
            assert child1.parity is not None, msg  # noqa: S101
            assert child2.parity is not None, msg  # noqa: S101
            return filter_parity_violating_ls(
                ls, parent.parity, child1.parity, child2.parity
            )
        return ls

    resonances = [particle_definitions[name] for name in resonance_names]
    chains: list[ThreeBodyDecayChain] = []
    for res in resonances:
        chains.extend(create_isobar(res))
    return __form_three_body_decay(chains)


def __form_three_body_decay(chains: Sequence[ThreeBodyDecayChain]) -> ThreeBodyDecay:
    if not chains:
        msg = "No decay chains were created"
        raise ValueError(msg)
    states: list[State] = [chains[0].initial_state, *chains[0].final_state]
    return ThreeBodyDecay(
        states={s.index: s for s in states},
        chains=tuple(chains),
    )


class ParameterBootstrap:
    """A wrapper for loading parameters from :download:`model-definitions.yaml </../data/model-definitions.yaml>`."""

    def __init__(
        self,
        filename: Path | str,
        decay: ThreeBodyDecay,
        model_id: int | ModelName = 0,
    ) -> None:
        particle_definitions = extract_particle_definitions(decay)
        symbolic_parameters = load_model_parameters_with_uncertainties(
            filename, decay, model_id, particle_definitions
        )
        self._parameters = {str(k): v for k, v in symbolic_parameters.items()}

    @property
    def values(self) -> dict[str, complex | int]:
        return {k: v.value for k, v in self._parameters.items()}

    @property
    def uncertainties(self) -> dict[str, complex | int]:
        return {k: v.uncertainty for k, v in self._parameters.items()}

    def create_distribution(
        self, sample_size: int, seed: int | None = None
    ) -> dict[str, np.ndarray]:
        return _smear_gaussian(
            parameter_values=self.values,
            parameter_uncertainties=self.uncertainties,
            size=sample_size,
            seed=seed,
        )


def load_model_parameters(
    filename: Path | str,
    decay: ThreeBodyDecay,
    model_id: int | ModelName = 0,
    particle_definitions: dict[str, Particle] | None = None,
) -> dict[sp.Indexed | sp.Symbol, complex]:
    parameters = load_model_parameters_with_uncertainties(
        filename, decay, model_id, particle_definitions
    )
    return {k: v.value for k, v in parameters.items()}


def load_model_parameters_with_uncertainties(
    filename: Path | str,
    decay: ThreeBodyDecay,
    model_id: int | ModelName = 0,
    particle_definitions: dict[str, Particle] | None = None,
) -> dict[sp.Indexed | sp.Symbol, MeasuredParameter]:
    with open(filename) as f:
        model_definitions = yaml.load(f, Loader=yaml.SafeLoader)
    if particle_definitions is None:
        msg = "Need to provide particle definitions"
        raise ValueError(msg)
    model_title = _find_model_title(model_definitions, model_id)
    min_ls = "LS couplings" not in model_title
    parameter_definitions = model_definitions[model_title]["parameters"]
    parameters = _to_symbol_value_mapping(
        parameter_definitions, min_ls, particle_definitions
    )
    decay_couplings = compute_decay_couplings(decay)
    parameters.update(decay_couplings)  # type:ignore[arg-type]  # pyright:ignore[reportCallIssue]
    return parameters  # type:ignore[return-value]


def _smear_gaussian(
    parameter_values: dict[str, complex],
    parameter_uncertainties: dict[str, complex],
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
    mean: complex,
    std: complex,
    size: int,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if isinstance(mean, complex) and isinstance(std, complex):
        return (
            rng.normal(mean.real, std.real, size)
            + rng.normal(mean.imag, std.imag, size) * 1j
        )
    if isinstance(mean, (float, int)) and isinstance(std, (float, int)):
        return rng.normal(mean, std, size)
    raise NotImplementedError


_T = TypeVar("_T")
_K = TypeVar("_K")
_V = TypeVar("_V")


def flip_production_coupling_signs(
    obj: _T, subsystem_names: Iterable[Literal["D", "K", "L"]]
) -> _T:
    if isinstance(obj, AmplitudeModel):
        return attrs.evolve(  # type:ignore[return-value]
            obj,
            parameter_defaults=_flip_signs(obj.parameter_defaults, subsystem_names),
        )
    if isinstance(obj, ParameterBootstrap):
        bootstrap = deepcopy(obj)
        bootstrap._parameters = _flip_signs(bootstrap._parameters, subsystem_names)  # type: ignore[reportPrivateUsage]  # noqa: SLF001
        return bootstrap  # type:ignore[return-value]
    if isinstance(obj, dict):
        return _flip_signs(obj, subsystem_names)  # type:ignore[return-value]
    raise NotImplementedError


def _flip_signs(
    parameters: dict[_K, _V], subsystem_names: Iterable[Literal["D", "K", "L"]]
) -> dict[_K, _V]:
    r"""Flip the signs of the production couplings for the given subsystems.

    >>> from pprint import pprint
    >>> parameters = {
    ...     R"\mathcal{H}^\mathrm{LS,production}[\Delta(1232), 1, 3/2]": +1,
    ...     R"\mathcal{H}^\mathrm{LS,production}[\Lambda(1405), 0, 1/2]": -1,
    ... }
    >>> flipped_parameters = _flip_signs(parameters, subsystem_names=["D"])
    >>> pprint(flipped_parameters)
    {'\\mathcal{H}^\\mathrm{LS,production}[\\Delta(1232), 1, 3/2]': -1,
     '\\mathcal{H}^\\mathrm{LS,production}[\\Lambda(1405), 0, 1/2]': -1}
    """
    pattern = r"\\mathrm{(LS,)?production}\[\\?" f"[{''.join(subsystem_names)}]"
    return {
        key: _flip(value) if re.search(pattern, str(key)) else value
        for key, value in parameters.items()
    }


def _flip(obj: _V) -> _V:
    if isinstance(obj, MeasuredParameter):
        return attrs.evolve(obj, value=_flip(obj.value))  # type:ignore[return-value]
    return -obj  # type:ignore[operator]


def compute_decay_couplings(
    decay: ThreeBodyDecay,
) -> dict[sp.Indexed, MeasuredParameter[int]]:
    H_dec = _get_coupling_base(helicity_basis=True, typ="decay")
    half = sp.Rational(1, 2)
    decay_couplings = {}
    for chain in decay.chains:
        R = _stringify(chain.resonance)
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
            msg = "Parity is not defined"
            assert chain.resonance.parity is not None, msg  # noqa: S101
            assert child1.parity is not None, msg  # noqa: S101
            assert child2.parity is not None, msg  # noqa: S101
            decay_couplings[coupling_pos] = 1
            decay_couplings[coupling_neg] = int(
                chain.resonance.parity
                * child1.parity
                * child2.parity
                * (-1) ** (chain.resonance.spin - child1.spin - child2.spin)
            )
    return {
        symbol: MeasuredParameter(value, hesse=0)
        for symbol, value in decay_couplings.items()
    }


def _to_symbol_value_mapping(
    parameter_dict: dict[str, str],
    min_ls: bool,
    particle_definitions: dict[str, Particle],
) -> dict[sp.Basic, MeasuredParameter]:
    key_to_value: dict[str, MeasuredParameter] = {}
    for key, str_value in parameter_dict.items():
        if key.startswith("Ar"):
            identifier = key[2:]
            str_imag = parameter_dict[f"Ai{identifier}"]
            key = f"A{identifier}"
            indexed_symbol: sp.Indexed = parameter_key_to_symbol(  # type:ignore[assignment]
                key, particle_definitions, min_ls
            )
            resonance_latex = str(indexed_symbol.indices[0])
            resonance, *_ = (
                p for p in particle_definitions.values() if p.latex == resonance_latex
            )
            if min_ls:
                conversion_factor = get_conversion_factor(resonance)
            else:
                _, L, S = indexed_symbol.indices
                conversion_factor = get_conversion_factor_ls(resonance, L, S)
            real = _to_value_with_uncertainty(str_value)
            imag = _to_value_with_uncertainty(str_imag)
            parameter = _form_complex_parameter(real, imag)
            key_to_value[key] = attrs.evolve(
                parameter,
                value=conversion_factor * parameter.value,
            )
        elif key.startswith("Ai"):
            continue
        else:
            key_to_value[key] = _to_value_with_uncertainty(str_value)
    return {
        parameter_key_to_symbol(key, particle_definitions, min_ls): value
        for key, value in key_to_value.items()
    }


def _to_value_with_uncertainty(str_value: str) -> MeasuredParameter[float]:
    """Convert a string to a `.MeasuredParameter` object.

    >>> _to_value_with_uncertainty("1.5 ± 0.2")
    MeasuredParameter(value=1.5, hesse=0.2, model=None, systematic=None)
    >>> par = _to_value_with_uncertainty("0.94 ± 0.042 ± 0.35 ± 0.04")
    >>> par
    MeasuredParameter(value=0.94, hesse=0.042, model=0.35, systematic=0.04)
    >>> par.uncertainty
    0.058
    """
    float_values = tuple(float(s) for s in str_value.split(" ± "))
    if len(float_values) == 2:
        return MeasuredParameter(
            value=float_values[0],
            hesse=float_values[1],
        )
    if len(float_values) == 4:
        return MeasuredParameter(
            value=float_values[0],
            hesse=float_values[1],
            model=float_values[2],
            systematic=float_values[3],
        )
    msg = f"Cannot convert '{str_value}' to {MeasuredParameter.__name__}"
    raise ValueError(msg)


def _form_complex_parameter(
    real: MeasuredParameter[float],
    imag: MeasuredParameter[float],
) -> MeasuredParameter[complex]:
    def convert_optional(real: float | None, imag: float | None) -> complex | None:
        if real is None or imag is None:
            return None
        return complex(real, imag)

    return MeasuredParameter(
        value=complex(real.value, imag.value),
        hesse=complex(real.hesse, imag.hesse),
        model=convert_optional(real.model, imag.model),
        systematic=convert_optional(real.systematic, imag.systematic),
    )


ParameterType = TypeVar("ParameterType", complex, float, int)
"""Template for the parameter type of a for `MeasuredParameter`."""


@frozen
class MeasuredParameter(Generic[ParameterType]):
    """Data structure for imported parameter values.

    `MeasuredParameter.value` and `~.MeasuredParameter.hesse` are taken from the
    `supplemental material <https://cds.cern.ch/record/2824328/files>`_, whereas
    `~.MeasuredParameter.model` and `~.MeasuredParameter.systematic` are taken from
    `Tables 8 and 9 <https://arxiv.org/pdf/2208.03262.pdf#page=21>`_ from the original
    LHCb paper :cite:`LHCb-PAPER-2022-002`.
    """

    value: ParameterType
    """Central value of the parameter as determined by a fit with Minuit."""
    hesse: ParameterType
    """Parameter uncertainty as determined by a fit with Minuit."""
    model: ParameterType | None = None
    """Systematic uncertainties from fit bootstrapping."""
    systematic: ParameterType | None = None
    """Systematic uncertainties from detector effects etc.."""

    @property
    def uncertainty(self) -> ParameterType:
        if self.systematic is None:
            return self.hesse
        if isinstance(self.value, (float, int)):
            return sqrt(self.hesse**2 + self.systematic**2)  # type:ignore[arg-type,return-value]
        return complex(
            sqrt(self.hesse.real**2 + self.systematic.real**2),
            sqrt(self.hesse.imag**2 + self.systematic.imag**2),
        )


def get_conversion_factor(resonance: Particle) -> Literal[-1, 1]:
    # https://github.com/ComPWA/polarimetry/issues/5#issue-1220525993
    half = sp.Rational(1, 2)
    assert resonance.parity is not None, "Parity is not defined"  # noqa: S101
    if resonance.name.startswith("D"):
        return int(-resonance.parity * (-1) ** (resonance.spin - half))  # type:ignore[return-value]
    if resonance.name.startswith("K"):
        return 1
    if resonance.name.startswith("L"):
        return int(-resonance.parity)  # type:ignore[return-value]
    msg = f"No conversion factor implemented for {resonance.name}"
    raise NotImplementedError(msg)


def get_conversion_factor_ls(
    resonance: Particle, L: sp.Rational, S: sp.Rational
) -> Literal[-1, 1]:
    # https://github.com/ComPWA/polarimetry/issues/192#issuecomment-1321892494
    if resonance.name.startswith("K") and resonance.spin == 0:
        return 1
    half = sp.Rational(1, 2)
    cg_flip_factor = int((-1) ** (L + S - half))
    return get_conversion_factor(resonance) * cg_flip_factor  # type:ignore[return-value]


def parameter_key_to_symbol(  # noqa: C901, PLR0911, PLR0912
    key: str,
    particle_definitions: dict[str, Particle],
    min_ls: bool = True,
) -> sp.Indexed | sp.Symbol:
    H_prod = _get_coupling_base(helicity_basis=min_ls, typ="production")
    half = sp.Rational(1, 2)
    if key.startswith("A"):
        # https://github.com/ComPWA/polarimetry/issues/5#issue-1220525993
        resonance_name = key[1:-1]
        resonance = particle_definitions[resonance_name]
        R = _stringify(resonance)
        subsystem_identifier = resonance.name[0]
        coupling_number = int(key[-1])
        if min_ls:
            # Helicity couplings
            if subsystem_identifier in {"D", "L"}:
                if coupling_number == 1:
                    return H_prod[R, -half, 0]
                if coupling_number == 2:
                    return H_prod[R, +half, 0]
            if subsystem_identifier == "K":
                if str(R) in {"K(700)", "K(1430)"}:
                    if coupling_number == 1:
                        return H_prod[R, 0, +half]
                    if coupling_number == 2:
                        return H_prod[R, 0, -half]
                else:
                    if coupling_number == 1:
                        return H_prod[R, 0, -half]
                    if coupling_number == 2:
                        return H_prod[R, -1, -half]
                    if coupling_number == 3:
                        return H_prod[R, +1, +half]
                    if coupling_number == 4:
                        return H_prod[R, 0, +half]
        else:
            # LS-couplings: supplemental material p.1 (https://cds.cern.ch/record/2824328/files)
            if subsystem_identifier in {"D", "L"}:
                if coupling_number == 1:
                    return H_prod[R, resonance.spin - half, resonance.spin]
                if coupling_number == 2:
                    return H_prod[R, resonance.spin + half, resonance.spin]
            if subsystem_identifier == "K":
                if resonance.spin == 0:  # "K(700)", "K(1430)"
                    if coupling_number == 1:
                        return H_prod[R, 0, half]
                    if coupling_number == 2:
                        return H_prod[R, 1, half]
                else:
                    if coupling_number == 1:
                        return H_prod[R, 0, half]
                    if coupling_number == 2:
                        return H_prod[R, 1, half]
                    if coupling_number == 3:
                        return H_prod[R, 1, 3 * half]
                    if coupling_number == 4:
                        return H_prod[R, 2, 3 * half]
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
        return sp.Symbol(Rf"\Gamma_{{{R} \to {K.latex} {p.latex}}}")
    if key.startswith("G2"):
        R = _stringify(key[2:])
        return sp.Symbol(Rf"\Gamma_{{{R} \to {π.latex} {Σ.latex}}}")
    if key.startswith("G"):
        R = _stringify(key[1:])
        return sp.Symbol(Rf"\Gamma_{{{R}}}")
    if key == "dLc":
        return sp.Symbol(R"R_{\Lambda_c}")
    msg = f'Cannot convert key "{key}" in model parameter JSON file to SymPy symbol'
    raise NotImplementedError(msg)


def _stringify(obj) -> Str:
    if isinstance(obj, Particle):
        return Str(obj.latex)
    return Str(f"{obj}")


def extract_particle_definitions(decay: ThreeBodyDecay) -> dict[str, Particle]:
    particles = {}

    def update_definitions(particle: Particle) -> None:
        particles[particle.name] = particle

    for chain in decay.chains:
        update_definitions(chain.parent)
        update_definitions(chain.resonance)
        update_definitions(chain.spectator)
        update_definitions(chain.decay_products[0])
        update_definitions(chain.decay_products[1])
    return particles

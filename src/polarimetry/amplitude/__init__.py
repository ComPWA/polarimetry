from __future__ import annotations

from functools import lru_cache
from itertools import product
from typing import Literal, Protocol

import sympy as sp
from ampform.sympy import PoolSum
from attrs import field, frozen
from sympy.core.symbol import Str
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.spin import Rotation as Wigner
from sympy.physics.quantum.spin import WignerD

from polarimetry.decay import (
    IsobarNode,
    Particle,
    ThreeBodyDecay,
    ThreeBodyDecayChain,
    get_decay_product_ids,
)
from polarimetry.spin import create_spin_range

from .angles import formulate_scattering_angle, formulate_zeta_angle


@frozen
class AmplitudeModel:
    decay: ThreeBodyDecay
    intensity: sp.Expr = sp.S.One
    amplitudes: dict[sp.Indexed, sp.Expr] = field(factory=dict)
    variables: dict[sp.Symbol, sp.Expr] = field(factory=dict)
    parameter_defaults: dict[sp.Symbol, float] = field(factory=dict)

    @property
    def full_expression(self) -> sp.Expr:
        return self.intensity.doit().xreplace(self.amplitudes)


class DalitzPlotDecompositionBuilder:
    def __init__(self, decay: ThreeBodyDecay, min_ls: bool = True) -> None:
        self.decay = decay
        self.dynamics_choices = DynamicsConfigurator(decay)
        self.min_ls = min_ls

    def formulate(
        self,
        reference_subsystem: Literal[1, 2, 3] = 1,
        cleanup_summations: bool = False,
    ) -> AmplitudeModel:
        helicity_symbols = sp.symbols("lambda:4", rational=True)
        allowed_helicities = {
            symbol: create_spin_range(self.decay.states[i].spin)
            for i, symbol in enumerate(helicity_symbols)
        }
        amplitude_definitions = {}
        angle_definitions = {}
        parameter_defaults = {}
        for args in product(*allowed_helicities.values()):
            for sub_system in [1, 2, 3]:
                chain_model = self.formulate_subsystem_amplitude(*args, sub_system)
                amplitude_definitions.update(chain_model.amplitudes)
                angle_definitions.update(chain_model.variables)
                parameter_defaults.update(chain_model.parameter_defaults)
        aligned_amp, zeta_defs = self.formulate_aligned_amplitude(
            *helicity_symbols, reference_subsystem
        )
        angle_definitions.update(zeta_defs)
        m0, m1, m2, m3 = sp.symbols("m:4", nonnegative=True)
        masses = {
            m0: self.decay.states[0].mass,
            m1: self.decay.states[1].mass,
            m2: self.decay.states[2].mass,
            m3: self.decay.states[3].mass,
        }
        parameter_defaults.update(masses)
        if cleanup_summations:
            aligned_amp = aligned_amp.cleanup()
        intensity = PoolSum(
            sp.Abs(aligned_amp) ** 2,
            *allowed_helicities.items(),
        )
        if cleanup_summations:
            intensity = intensity.cleanup()
        return AmplitudeModel(
            decay=self.decay,
            intensity=PoolSum(
                sp.Abs(aligned_amp) ** 2,
                *allowed_helicities.items(),
            ),
            amplitudes=amplitude_definitions,
            variables=angle_definitions,
            parameter_defaults=parameter_defaults,
        )

    def formulate_subsystem_amplitude(
        self,
        λ0: sp.Rational,
        λ1: sp.Rational,
        λ2: sp.Rational,
        λ3: sp.Rational,
        subsystem_id: Literal[1, 2, 3],
    ) -> AmplitudeModel:
        k = subsystem_id
        i, j = get_decay_product_ids(subsystem_id)
        θij, θij_expr = formulate_scattering_angle(i, j)
        λ = λ0, λ1, λ2, λ3
        spin = [
            self.decay.initial_state.spin,
            self.decay.final_state[1].spin,
            self.decay.final_state[2].spin,
            self.decay.final_state[3].spin,
        ]
        H_prod = get_indexed_base("production", self.min_ls)
        H_dec = get_indexed_base("decay")
        λR = sp.Symbol(R"\lambda_R", rational=True)
        terms = []
        parameter_defaults = {}
        for chain in self.decay.get_subsystem(subsystem_id).chains:
            formulate_dynamics = self.dynamics_choices.get_builder(chain.resonance.name)
            dynamics, new_parameters = formulate_dynamics(chain)
            parameter_defaults.update(new_parameters)
            R = Str(chain.resonance.name)
            resonance_spin = sp.Rational(chain.resonance.spin)
            resonance_helicities = create_spin_range(resonance_spin)
            for λR_val in resonance_helicities:
                if λ[0] == λR_val - λ[k]:  # Kronecker delta
                    if self.min_ls:
                        parameter_defaults[H_prod[R, λR_val, λ[k]]] = 1
                    parameter_defaults[H_dec[R, λ[i], λ[j]]] = 1
            sub_amp_expr = (
                sp.KroneckerDelta(λ[0], λR - λ[k])
                * (-1) ** (spin[k] - λ[k])
                * dynamics
                * Wigner.d(resonance_spin, λR, λ[i] - λ[j], θij)
                * H_dec[R, λ[i], λ[j]]
                * (-1) ** (spin[j] - λ[j])
            )
            if self.min_ls:
                sub_amp_expr *= H_prod[R, λR, λ[k]]
            else:
                production_isobar = chain.decay
                resonance_isobar = chain.decay.child1
                assert production_isobar.interaction is not None  # noqa: S101
                assert resonance_isobar.interaction is not None  # noqa: S101
                sub_amp_expr *= H_prod[
                    R,
                    production_isobar.interaction.L,
                    production_isobar.interaction.S,
                ]
                sub_amp_expr *= _formulate_clebsch_gordan_factor(
                    production_isobar,
                    child1_helicity=λR,
                    child2_helicity=λ[k],
                )
            sub_amp = PoolSum(
                sub_amp_expr,
                (λR, resonance_helicities),
            )
            terms.append(sub_amp)
        A = _generate_amplitude_index_bases()
        amp_symbol = A[subsystem_id][λ0, λ1, λ2, λ3]
        amp_expr = sp.Add(*terms)
        return AmplitudeModel(
            decay=self.decay,
            intensity=sp.Abs(amp_symbol) ** 2,
            amplitudes={amp_symbol: amp_expr},
            variables={θij: θij_expr},
            parameter_defaults=parameter_defaults,
        )

    def formulate_aligned_amplitude(
        self,
        λ0: sp.Rational | sp.Symbol,
        λ1: sp.Rational | sp.Symbol,
        λ2: sp.Rational | sp.Symbol,
        λ3: sp.Rational | sp.Symbol,
        reference_subsystem: Literal[1, 2, 3] = 1,
    ) -> tuple[PoolSum, dict[sp.Symbol, sp.Expr]]:
        wigner_generator = _AlignmentWignerGenerator(reference_subsystem)
        _λ0, _λ1, _λ2, _λ3 = sp.symbols(R"\lambda_(0:4)^{\prime}", rational=True)
        j0, j1, j2, j3 = (self.decay.states[i].spin for i in sorted(self.decay.states))
        A = _generate_amplitude_index_bases()
        amp_expr = PoolSum(
            A[1][_λ0, _λ1, _λ2, _λ3]
            * wigner_generator(j0, λ0, _λ0, rotated_state=0, aligned_subsystem=1)
            * wigner_generator(j1, _λ1, λ1, rotated_state=1, aligned_subsystem=1)
            * wigner_generator(j2, _λ2, λ2, rotated_state=2, aligned_subsystem=1)
            * wigner_generator(j3, _λ3, λ3, rotated_state=3, aligned_subsystem=1)
            + A[2][_λ0, _λ1, _λ2, _λ3]
            * wigner_generator(j0, λ0, _λ0, rotated_state=0, aligned_subsystem=2)
            * wigner_generator(j1, _λ1, λ1, rotated_state=1, aligned_subsystem=2)
            * wigner_generator(j2, _λ2, λ2, rotated_state=2, aligned_subsystem=2)
            * wigner_generator(j3, _λ3, λ3, rotated_state=3, aligned_subsystem=2)
            + A[3][_λ0, _λ1, _λ2, _λ3]
            * wigner_generator(j0, λ0, _λ0, rotated_state=0, aligned_subsystem=3)
            * wigner_generator(j1, _λ1, λ1, rotated_state=1, aligned_subsystem=3)
            * wigner_generator(j2, _λ2, λ2, rotated_state=2, aligned_subsystem=3)
            * wigner_generator(j3, _λ3, λ3, rotated_state=3, aligned_subsystem=3),
            (_λ0, create_spin_range(j0)),
            (_λ1, create_spin_range(j1)),
            (_λ2, create_spin_range(j2)),
            (_λ3, create_spin_range(j3)),
        )
        return amp_expr, wigner_generator.angle_definitions


def get_indexed_base(
    typ: Literal["production", "decay"], min_ls: bool = True
) -> sp.IndexedBase:
    """Get a basis to generate coupling symbols for the production or decay node."""
    if typ == "decay":
        return sp.IndexedBase(R"\mathcal{H}^\mathrm{decay}")
    if min_ls:
        return sp.IndexedBase(R"\mathcal{H}^\mathrm{production}")
    return sp.IndexedBase(R"\mathcal{H}^\mathrm{LS,production}")


@lru_cache(maxsize=None)
def _generate_amplitude_index_bases() -> dict[Literal[1, 2, 3], sp.IndexedBase]:
    return dict(enumerate(sp.symbols(R"A^(1:4)", cls=sp.IndexedBase), 1))


class _AlignmentWignerGenerator:
    def __init__(self, reference_subsystem: Literal[1, 2, 3] = 1) -> None:
        self.angle_definitions: dict[sp.Symbol, sp.acos] = {}
        self.reference_subsystem = reference_subsystem

    def __call__(
        self,
        j: sp.Rational,
        m: sp.Rational,
        m_prime: sp.Rational,
        rotated_state: int,
        aligned_subsystem: int,
    ) -> sp.Rational | WignerD:
        if j == 0:
            return sp.Rational(1)
        zeta, zeta_expr = formulate_zeta_angle(
            rotated_state, aligned_subsystem, self.reference_subsystem
        )
        self.angle_definitions[zeta] = zeta_expr
        return Wigner.d(j, m, m_prime, zeta)


class DynamicsConfigurator:
    def __init__(self, decay: ThreeBodyDecay) -> None:
        self.__decay = decay
        self.__dynamics_builders: dict[ThreeBodyDecayChain, DynamicsBuilder] = {}

    def register_builder(self, identifier, builder: DynamicsBuilder) -> None:
        chain = self.__get_chain(identifier)
        self.__dynamics_builders[chain] = builder

    def get_builder(self, identifier) -> DynamicsBuilder:
        chain = self.__get_chain(identifier)
        return self.__dynamics_builders[chain]

    def __get_chain(self, identifier) -> ThreeBodyDecayChain:
        if isinstance(identifier, ThreeBodyDecayChain):
            chain = identifier
            if chain not in set(self.__decay.chains):
                msg = f"Decay does not have chain with resonance {chain.resonance.name}"
                raise ValueError(msg)
            return chain
        if isinstance(identifier, str):
            return self.__decay.find_chain(identifier)
        msg = f"Cannot get decay chain for identifier type {type(identifier)}"
        raise NotImplementedError(msg)

    @property
    def decay(self) -> ThreeBodyDecay:
        return self.__decay


class DynamicsBuilder(Protocol):
    def __call__(
        self, decay_chain: ThreeBodyDecayChain
    ) -> tuple[sp.Expr, dict[sp.Symbol, float]]: ...


def simplify_latex_rendering() -> None:
    """Improve LaTeX rendering of an `~sympy.tensor.indexed.Indexed` object."""

    def _print_Indexed_latex(self, printer, *args):  # noqa: N802
        base = printer._print(self.base)
        indices = ", ".join(map(printer._print, self.indices))
        return f"{base}_{{{indices}}}"

    sp.Indexed._latex = _print_Indexed_latex


def _formulate_clebsch_gordan_factor(
    isobar: IsobarNode,
    child1_helicity: sp.Rational | sp.Symbol,
    child2_helicity: sp.Rational | sp.Symbol,
) -> sp.Expr:
    if isobar.interaction is None:
        msg = "Cannot formulate amplitude model in LS-basis if LS-couplings are missing"
        raise ValueError(msg)
    # https://github.com/ComPWA/ampform/blob/65b4efa/src/ampform/helicity/__init__.py#L785-L802
    # and supplementary material p.1 (https://cds.cern.ch/record/2824328/files)
    cg_ss = CG(
        j1=_get_particle(isobar.child1).spin,
        m1=child1_helicity,
        j2=_get_particle(isobar.child2).spin,
        m2=-child2_helicity,
        j3=isobar.interaction.S,
        m3=child1_helicity - child2_helicity,
    )
    cg_ll = CG(
        j1=isobar.interaction.L,
        m1=0,
        j2=isobar.interaction.S,
        m2=child1_helicity - child2_helicity,
        j3=isobar.parent.spin,
        m3=child1_helicity - child2_helicity,
    )
    sqrt_factor = sp.sqrt((2 * isobar.interaction.L + 1) / (2 * isobar.parent.spin + 1))
    return sqrt_factor * cg_ll * cg_ss


def _get_particle(isobar: IsobarNode | Particle) -> Particle:
    if isinstance(isobar, IsobarNode):
        return isobar.parent
    return isobar

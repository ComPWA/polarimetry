from __future__ import annotations

import sys
from itertools import product

import sympy as sp
from ampform.sympy import PoolSum
from attrs import field, frozen
from sympy.core.symbol import Str
from sympy.physics.quantum.spin import Rotation as Wigner

from polarization.decay import (
    ThreeBodyDecay,
    ThreeBodyDecayChain,
    get_decay_product_ids,
)
from polarization.spin import create_spin_range

from .angles import formulate_scattering_angle, formulate_zeta_angle

if sys.version_info < (3, 8):
    from typing_extensions import Literal, Protocol
else:
    from typing import Literal, Protocol


def _print_Indexed_latex(self, printer, *args):
    """Improved LaTeX rendering of a `sympy.Indexed` object."""
    base = printer._print(self.base)
    indices = ", ".join(map(printer._print, self.indices))
    return f"{base}_{{{indices}}}"


sp.Indexed._latex = _print_Indexed_latex


A1, A2, A3 = sp.symbols(R"A^(1:4)", cls=sp.IndexedBase)
A = {
    1: A1,
    2: A2,
    3: A3,
}


@frozen
class AmplitudeModel:
    intensity: sp.Expr = sp.S.One
    amplitudes: dict[sp.Indexed, sp.Expr] = field(factory=dict)
    variables: dict[sp.Symbol, sp.Expr] = field(factory=dict)
    parameter_defaults: dict[sp.Symbol, float] = field(factory=dict)

    @property
    def full_expression(self) -> sp.Expr:
        return self.intensity.doit().xreplace(self.amplitudes)


class DalitzPlotDecompositionBuilder:
    def __init__(self, decay: ThreeBodyDecay) -> None:
        self.decay = decay
        self.dynamics_choices = DynamicsConfigurator(decay)

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
        H_prod = sp.IndexedBase(R"\mathcal{H}^\mathrm{production}")
        H_dec = sp.IndexedBase(R"\mathcal{H}^\mathrm{decay}")
        λR = sp.Symbol(R"\lambda_R", rational=True)
        terms = []
        parameter_defaults = {}
        for chain in self.decay.get_subsystem(subsystem_id).chains:
            formulate_dynamics = self.dynamics_choices.get_builder(chain.resonance.name)
            dynamics, new_parameters = formulate_dynamics(chain)
            parameter_defaults.update(new_parameters)
            R = Str(chain.resonance.latex)
            resonance_spin = sp.Rational(chain.resonance.spin)
            resonance_helicities = create_spin_range(resonance_spin)
            for λR_val in resonance_helicities:
                if λ[0] == λR_val - λ[k]:  # Kronecker delta
                    parameter_defaults[H_prod[R, λR_val, λ[k]]] = 1
                    parameter_defaults[H_dec[R, λ[i], λ[j]]] = 1
            sub_amp = PoolSum(
                sp.KroneckerDelta(λ[0], λR - λ[k])
                * H_prod[R, λR, λ[k]]
                * (-1) ** (spin[k] - λ[k])
                * dynamics
                * Wigner.d(resonance_spin, λR, λ[i] - λ[j], θij)
                * H_dec[R, λ[i], λ[j]]
                * (-1) ** (spin[j] - λ[j]),
                (λR, resonance_helicities),
            )
            terms.append(sub_amp)
        amp_symbol = A[subsystem_id][λ0, λ1, λ2, λ3]
        amp_expr = sp.Add(*terms)
        return AmplitudeModel(
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
        zeta01, zeta01_expr = formulate_zeta_angle(0, 1, reference_subsystem)
        zeta02, zeta02_expr = formulate_zeta_angle(0, 2, reference_subsystem)
        zeta03, zeta03_expr = formulate_zeta_angle(0, 3, reference_subsystem)
        zeta11, zeta11_expr = formulate_zeta_angle(1, 1, reference_subsystem)
        zeta21, zeta21_expr = formulate_zeta_angle(2, 1, reference_subsystem)
        zeta31, zeta31_expr = formulate_zeta_angle(3, 1, reference_subsystem)
        zeta12, zeta12_expr = formulate_zeta_angle(1, 2, reference_subsystem)
        zeta22, zeta22_expr = formulate_zeta_angle(2, 2, reference_subsystem)
        zeta32, zeta32_expr = formulate_zeta_angle(3, 2, reference_subsystem)
        zeta13, zeta13_expr = formulate_zeta_angle(1, 3, reference_subsystem)
        zeta23, zeta23_expr = formulate_zeta_angle(2, 3, reference_subsystem)
        zeta33, zeta33_expr = formulate_zeta_angle(3, 3, reference_subsystem)
        symbol_definitions = {
            zeta01: zeta01_expr,
            zeta02: zeta02_expr,
            zeta03: zeta03_expr,
            zeta11: zeta11_expr,
            zeta21: zeta21_expr,
            zeta31: zeta31_expr,
            zeta12: zeta12_expr,
            zeta22: zeta22_expr,
            zeta32: zeta32_expr,
            zeta13: zeta13_expr,
            zeta23: zeta23_expr,
            zeta33: zeta33_expr,
        }
        _λ0, _λ1, _λ2, _λ3 = sp.symbols(R"\lambda_(0:4)^{\prime}", rational=True)
        j0, j1, j2, j3 = (self.decay.states[i].spin for i in sorted(self.decay.states))
        amp_expr = PoolSum(
            A1[_λ0, _λ1, _λ2, _λ3]
            * Wigner.d(j0, λ0, _λ0, zeta01)
            * Wigner.d(j1, _λ1, λ1, zeta11)
            * Wigner.d(j2, _λ2, λ2, zeta21)
            * Wigner.d(j3, _λ3, λ3, zeta31)
            + A2[_λ0, _λ1, _λ2, _λ3]
            * Wigner.d(j0, λ0, _λ0, zeta02)
            * Wigner.d(j1, _λ1, λ1, zeta12)
            * Wigner.d(j2, _λ2, λ2, zeta22)
            * Wigner.d(j3, _λ3, λ3, zeta32)
            + A3[_λ0, _λ1, _λ2, _λ3]
            * Wigner.d(j0, λ0, _λ0, zeta03)
            * Wigner.d(j1, _λ1, λ1, zeta13)
            * Wigner.d(j2, _λ2, λ2, zeta23)
            * Wigner.d(j3, _λ3, λ3, zeta33),
            (_λ0, create_spin_range(j0)),
            (_λ1, create_spin_range(j1)),
            (_λ2, create_spin_range(j2)),
            (_λ3, create_spin_range(j3)),
        )
        return amp_expr, symbol_definitions

    def formulate(self, alignment_subsystem: Literal[1, 2, 3] = 1) -> AmplitudeModel:
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
            *helicity_symbols, alignment_subsystem
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
        return AmplitudeModel(
            intensity=PoolSum(
                sp.Abs(aligned_amp) ** 2,
                *allowed_helicities.items(),
            ),
            amplitudes=amplitude_definitions,
            variables=angle_definitions,
            parameter_defaults=parameter_defaults,
        )


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
                raise ValueError(
                    f"Decay does not have chain with resonance {chain.resonance.name}"
                )
            return chain
        if isinstance(identifier, str):
            return self.__decay.find_chain(identifier)
        raise NotImplementedError(
            f"Cannot get decay chain for identifier type {type(identifier)}"
        )

    @property
    def decay(self) -> ThreeBodyDecay:
        return self.__decay


class DynamicsBuilder(Protocol):
    def __call__(
        self, decay_chain: ThreeBodyDecayChain
    ) -> tuple[sp.Expr, dict[sp.Symbol, float]]:
        ...

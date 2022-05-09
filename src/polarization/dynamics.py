# pyright: reportPrivateUsage=false
from __future__ import annotations

import sys

import sympy as sp
from ampform.sympy import (
    UnevaluatedExpression,
    create_expression,
    implement_doit_method,
    make_commutative,
)
from attrs import frozen

from polarization.decay import Particle

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal


@make_commutative
@implement_doit_method
class Källén(UnevaluatedExpression):
    def __new__(cls, x, y, z, **hints):
        return create_expression(cls, x, y, z, **hints)

    def evaluate(self) -> sp.Expr:
        x, y, z = self.args
        return x**2 + y**2 + z**2 - 2 * x * y - 2 * y * z - 2 * z * x

    def _latex(self, printer, *args):
        x, y, z = map(printer._print, self.args)
        return Rf"\lambda\left({x}, {y}, {z}\right)"


@make_commutative
@implement_doit_method
class P(UnevaluatedExpression):
    def __new__(cls, s, mi, mj, **hints):
        return create_expression(cls, s, mi, mj, **hints)

    def evaluate(self):
        s, mi, mj = self.args
        return sp.sqrt(Källén(s, mi**2, mj**2)) / (2 * sp.sqrt(s))

    def _latex(self, printer, *args):
        s = printer._print(self.args[0])
        return Rf"p_{{{s}}}"


@make_commutative
@implement_doit_method
class Q(UnevaluatedExpression):
    def __new__(cls, s, m0, mk, **hints):
        return create_expression(cls, s, m0, mk, **hints)

    def evaluate(self):
        s, m0, mk = self.args
        return sp.sqrt(Källén(s, m0**2, mk**2)) / (2 * m0)  # <-- not s!

    def _latex(self, printer, *args):
        s = printer._print(self.args[0])
        return Rf"q_{{{s}}}"


@make_commutative
@implement_doit_method
class RelativisticBreitWigner(UnevaluatedExpression):
    def __new__(cls, s, m0, Γ0, m1, m2, l_R, l_Λc, R):
        return create_expression(cls, s, m0, Γ0, m1, m2, l_R, l_Λc, R)

    def evaluate(self):
        s, m0, Γ0, m1, m2, l_R, l_Λc, R = self.args
        q = Q(s, m1, m2)
        q0 = Q(m0**2, m1, m2)
        p = P(s, m1, m2)
        p0 = P(m0**2, m1, m2)
        width = EnergyDependentWidth(s, m0, Γ0, m1, m2, l_R, R)
        return sp.Mul(
            (q / q0) ** l_Λc,
            BlattWeisskopf(q * R, l_Λc) / BlattWeisskopf(q0 * R, l_Λc),
            1 / (m0**2 - s - sp.I * m0 * width),
            (p / p0) ** l_R,
            BlattWeisskopf(p * R, l_R) / BlattWeisskopf(p0 * R, l_R),
            evaluate=False,
        )

    def _latex(self, printer, *args) -> str:
        s = printer._print(self.args[0])
        return Rf"\mathcal{{R}}\left({s}\right)"


@make_commutative
@implement_doit_method
class EnergyDependentWidth(UnevaluatedExpression):
    def __new__(cls, s, m0, Γ0, m1, m2, L, R):
        return create_expression(cls, s, m0, Γ0, m1, m2, L, R)

    def evaluate(self):
        s, m0, Γ0, m1, m2, L, R = self.args
        p = P(s, m1, m2)
        p0 = P(m0**2, m1, m2)
        ff = BlattWeisskopf(p * R, L) ** 2
        ff0 = BlattWeisskopf(p0 * R, L) ** 2
        return sp.Mul(
            Γ0,
            (p / p0) ** (2 * L + 1),
            m0 / sp.sqrt(s),
            ff / ff0,
            evaluate=False,
        )

    def _latex(self, printer, *args) -> str:
        s = printer._print(self.args[0])
        return Rf"\Gamma\left({s}\right)"


@make_commutative
@implement_doit_method
class BlattWeisskopf(UnevaluatedExpression):
    def __new__(cls, z, L, **hints):
        return create_expression(cls, z, L, **hints)

    def evaluate(self):
        z, L = self.args
        cases = {
            0: 1,
            1: 1 / (1 + z**2),
            2: 1 / (9 + 3 * z**2 + z**4),
        }
        return sp.Piecewise(
            *[(sp.sqrt(expr), sp.Eq(L, l_val)) for l_val, expr in cases.items()]
        )

    def _latex(self, printer, *args):
        z, L = map(printer._print, self.args)
        return Rf"F_{{{L}}}\left({z}\right)"


@frozen
class Resonance(Particle):
    mass_range: tuple[float, float]
    width_range: tuple[float, float]
    lineshape: Literal["BreitWignerMinL", "BuggBreitWignerMinL", "Flatte1405"]

    @property
    def mass(self) -> float:
        return _compute_average(self.mass_range)

    @property
    def width(self) -> float:
        return _compute_average(self.width_range)


def _compute_average(range_def: float | tuple[float, float]) -> float:
    _min, _max = range_def
    return (_max + _min) / 2

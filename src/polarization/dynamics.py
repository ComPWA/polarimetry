# pyright: reportPrivateUsage=false
from __future__ import annotations

import sympy as sp
from ampform.sympy import (
    UnevaluatedExpression,
    create_expression,
    implement_doit_method,
    make_commutative,
)


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
        s = printer._print(self.args[0], *args)
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
class BuggBreitWigner(UnevaluatedExpression):
    def __new__(cls, s, m0, Γ0, m1, m2, γ):
        return create_expression(cls, s, m0, Γ0, m1, m2, γ)

    def evaluate(self):
        s, m0, Γ0, m1, m2, γ = self.args
        s_A = m1**2 - m2**2
        g_squared = sp.Mul(
            (s - s_A) / (m0**2 - s_A),
            Γ0 * sp.exp(-γ * s),
            evaluate=False,
        )
        rho = 2 * P(s, m1, m2) / s
        return 1 / (m0**2 - s - sp.I * g_squared * rho)

    def _latex(self, printer, *args) -> str:
        s = printer._print(self.args[0], *args)
        return Rf"\mathcal{{R}}_\mathrm{{Bugg}}\left({s}\right)"


@make_commutative
@implement_doit_method
class FlattéSWave(UnevaluatedExpression):
    # https://github.com/redeboer/polarization-sensitivity/blob/34f5330/julia/notebooks/model0.jl#L151-L161
    def __new__(cls, s, m0, Γ0, masses1, masses2):
        return create_expression(cls, s, m0, Γ0, masses1, masses2)

    def evaluate(self):
        s, m0, Γ0, (ma1, mb1), (ma2, mb2) = self.args
        p = P(s, ma1, mb1)
        p0 = P(m0**2, ma2, mb2)
        q = P(s, ma2, mb2)
        q0 = P(m0**2, ma2, mb2)
        Γ1 = Γ0 * (p / p0) * m0 / sp.sqrt(s)
        Γ2 = Γ0 * (q / q0) * m0 / sp.sqrt(s)
        Γ = Γ1 + Γ2
        return 1 / (m0**2 - s - sp.I * m0 * Γ)

    def _latex(self, printer, *args) -> str:
        s = printer._print(self.args[0])
        return Rf"\mathcal{{R}}_\mathrm{{Flatté}}\left({s}\right)"


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

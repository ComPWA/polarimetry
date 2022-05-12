"""Input-output functions for `ampform` and `sympy` objects.

Functions in this module are registered with :func:`functools.singledispatch` and can be
extended as follows:

>>> from polarization.io import as_latex
>>> @as_latex.register(int)
... def _(obj: int) -> str:
...     return "my custom rendering"
>>> as_latex(1)
'my custom rendering'
>>> as_latex(3.4 - 2j)
'3.4-2i'

This code originates from `ComPWA/ampform#280
<https://github.com/ComPWA/ampform/pull/280>`_.
"""
from __future__ import annotations

from collections import abc
from functools import singledispatch
from typing import Iterable, Mapping

import qrules
import sympy as sp
from ampform.sympy import UnevaluatedExpression
from IPython.display import Math, display

from polarization.decay import IsobarNode, Particle, Resonance, ThreeBodyDecay


@singledispatch
def as_latex(obj, **kwargs) -> str:
    """Render objects as a LaTeX `str`.

    The resulting `str` can for instance be given to `IPython.display.Math`.

    Optional keywords:

    - `only_jp`: Render a `Particle` as :math:`J^P` value (spin-parity) only.
    - `with_jp`: Render a `Particle` with value :math:`J^P` value.
    """
    return str(obj, **kwargs)


@as_latex.register(complex)
def _(obj: complex, **kwargs) -> str:
    real = __downcast(obj.real)
    imag = __downcast(obj.imag)
    plus = "+" if imag >= 0 else ""
    return f"{real}{plus}{imag}i"


def __downcast(obj: float) -> float | int:
    if obj.is_integer():
        return int(obj)
    return obj


@as_latex.register(sp.Basic)
def _(obj: sp.Basic, **kwargs) -> str:
    return sp.latex(obj)


@as_latex.register(abc.Mapping)
def _(obj: Mapping, **kwargs) -> str:
    if len(obj) == 0:
        raise ValueError("Need at least one dictionary item")
    latex = R"\begin{array}{rcl}" + "\n"
    for lhs, rhs in obj.items():
        latex += Rf"  {as_latex(lhs, **kwargs)} &=& {as_latex(rhs, **kwargs)} \\" + "\n"
    latex += R"\end{array}"
    return latex


@as_latex.register(abc.Iterable)
def _(obj: Iterable, **kwargs) -> str:
    obj = list(obj)
    if len(obj) == 0:
        raise ValueError("Need at least one item to render as LaTeX")
    latex = R"\begin{array}{c}" + "\n"
    for item in obj:
        item_latex = as_latex(item, **kwargs)
        latex += Rf"  {item_latex} \\" + "\n"
    latex += R"\end{array}"
    return latex


@as_latex.register(IsobarNode)
def _(obj: IsobarNode, **kwargs) -> str:
    def render_arrow(node: IsobarNode) -> str:
        if node.interaction is None:
            return R"\to"
        return Rf"\xrightarrow[S={node.interaction.S}]{{L={node.interaction.L}}}"

    parent = as_latex(obj.parent, **kwargs)
    to = render_arrow(obj)
    child1 = as_latex(obj.child1, **kwargs)
    child2 = as_latex(obj.child2, **kwargs)
    return Rf"{parent} {to} {child1} {child2}"


@as_latex.register(ThreeBodyDecay)
def _(obj: ThreeBodyDecay, **kwargs) -> str:
    return as_latex(obj.decay)


@as_latex.register(Particle)
def _(obj: Particle, with_jp: bool = False, only_jp: bool = False, **kwargs) -> str:
    if only_jp:
        return _render_jp(obj)
    if with_jp:
        jp = _render_jp(obj)
        return Rf"{obj.latex}\left[{jp}\right]"
    return obj.latex


def _render_jp(particle: Particle) -> str:
    parity = "-1" if particle.parity < 0 else "+1"
    if particle.spin.denominator == 1:
        spin = sp.latex(particle.spin)
    else:
        spin = Rf"\frac{{{particle.spin.numerator}}}{{{particle.spin.denominator}}}"
    return f"{spin}^{{{parity}}}"


def display_latex(obj) -> None:
    latex = as_latex(obj)
    display(Math(latex))


def display_doit(
    expr: UnevaluatedExpression, deep=False, terms_per_line: int | None = None
) -> None:
    if terms_per_line is None:
        latex = as_latex({expr: expr.doit(deep=deep)})
    else:
        latex = sp.multiline_latex(
            lhs=expr,
            rhs=expr.doit(deep=deep),
            terms_per_line=terms_per_line,
            environment="eqnarray",
        )
    display(Math(latex))


@singledispatch
def from_qrules(obj):
    raise NotImplementedError(
        f"Cannot import QRules object of type {type(obj).__name__}"
    )


@from_qrules.register(qrules.particle.Particle)
def _(obj: qrules.particle.Particle) -> Resonance:
    if obj.parity is None:
        raise ValueError(f"Particle {obj.name} as no parity")
    return Resonance(
        name=obj.name,
        latex=obj.latex,
        spin=obj.spin,
        parity=obj.parity,
        mass_range=(obj.mass, obj.mass),
        width_range=(obj.width, obj.width),
        lineshape="BreitWignerMinL",
    )

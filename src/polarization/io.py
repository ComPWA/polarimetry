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

import sys
from collections import abc
from functools import singledispatch
from typing import Iterable, Mapping

import sympy as sp

from polarization.decay import IsobarNode, Particle
from polarization.dynamics import Resonance

if sys.version_info < (3, 8):
    from typing_extensions import Literal, TypedDict
else:
    from typing import Literal, TypedDict


@singledispatch
def as_latex(obj) -> str:
    """Render objects as a LaTeX `str`.

    The resulting `str` can for instance be given to `IPython.display.Math`.

    Optional keywords:

    - `render_jp`: Render a `Particle` as :math:`J^P` (spin-parity).
    """
    return str(obj)


@as_latex.register(complex)
def _(obj: complex) -> str:
    real = __downcast(obj.real)
    imag = __downcast(obj.imag)
    plus = "+" if imag >= 0 else ""
    return f"{real}{plus}{imag}i"


def __downcast(obj: float) -> float | int:
    if obj.is_integer():
        return int(obj)
    return obj


@as_latex.register(sp.Basic)
def _(obj: sp.Basic) -> str:
    return sp.latex(obj)


@as_latex.register(abc.Mapping)
def _(obj: Mapping) -> str:
    if len(obj) == 0:
        raise ValueError("Need at least one dictionary item")
    latex = R"\begin{array}{rcl}" + "\n"
    for lhs, rhs in obj.items():
        latex += Rf"  {as_latex(lhs)} &=& {as_latex(rhs)} \\" + "\n"
    latex += R"\end{array}"
    return latex


@as_latex.register(abc.Iterable)
def _(obj: Iterable) -> str:
    obj = list(obj)
    if len(obj) == 0:
        raise ValueError("Need at least one item to render as LaTeX")
    latex = R"\begin{array}{c}" + "\n"
    for item in map(as_latex, obj):
        latex += Rf"  {item} \\" + "\n"
    latex += R"\end{array}"
    return latex


@as_latex.register(IsobarNode)
def _(obj: IsobarNode, render_jp: bool = False) -> str:
    def render_arrow(node: IsobarNode) -> str:
        if node.interaction is None:
            return R"\to"
        return Rf"\xrightarrow[S={node.interaction.S}]{{L={node.interaction.L}}}"

    parent = as_latex(obj.parent, render_jp)
    to = render_arrow(obj)
    child1 = as_latex(obj.child1, render_jp)
    child2 = as_latex(obj.child2, render_jp)
    return Rf"{parent} {to} {child1} {child2}"


@as_latex.register(Particle)
def _(obj: Particle, render_jp: bool = False) -> str:
    if render_jp:
        parity = "-1" if obj.parity < 0 else "+1"
        return f"{{{obj.spin}}}^{{{parity}}}"
    return obj.name


def to_resonance_dict(definition: dict[str, ResonanceJSON]) -> dict[str, Resonance]:
    return {
        name: to_resonance(name, resonance_def)
        for name, resonance_def in definition.items()
    }


def to_resonance(name: str, definition: ResonanceJSON) -> Resonance:
    spin, parity = _to_jp_pair(definition["jp"])
    return Resonance(
        Particle(name, spin, parity),
        mass_range=_to_float_range(definition["mass"]),
        width_range=_to_float_range(definition["width"]),
        lineshape=definition["lineshape"],
    )


def _to_float_range(input_str: str) -> tuple[float, float]:
    """
    >>> _convert_mass_string("1405.1")
    (1405.1, 1405.1)
    >>> _convert_mass_string("1900-2100")
    (1900.0, 2100.0)
    """
    if "-" in input_str:
        _min, _max, *_ = map(float, input_str.split("-"))
    else:
        _min = _max = float(input_str)
    return _min, _max


def _to_jp_pair(input_str: str) -> tuple[sp.Rational, int]:
    """
    >>> _convert_jp_string("3/2^-")
    (3/2, -1)
    >>> _convert_jp_string("0^+")
    (0, 1)
    """
    spin, parity_sign = input_str.split("^")
    return sp.Rational(spin), int(f"{parity_sign}1")


class ResonanceJSON(TypedDict):
    jp: str
    mass: str
    width: str
    lineshape: Literal["BreitWignerMinL", "BuggBreitWignerMinL", "Flatte1405"]

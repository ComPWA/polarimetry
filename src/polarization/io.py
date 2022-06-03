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
from typing import Iterable, Mapping, Sequence

import sympy as sp
from ampform.sympy import UnevaluatedExpression
from IPython.display import Math, display

from polarization.decay import IsobarNode, Particle, ThreeBodyDecayChain


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


@as_latex.register(ThreeBodyDecayChain)
def _(obj: ThreeBodyDecayChain, **kwargs) -> str:
    return as_latex(obj.decay, **kwargs)


@as_latex.register(Particle)
def _(obj: Particle, with_jp: bool = False, only_jp: bool = False, **kwargs) -> str:
    if only_jp:
        return _render_jp(obj)
    if with_jp:
        jp = _render_jp(obj)
        return Rf"{obj.latex}\left[{jp}\right]"
    return obj.latex


def _render_jp(particle: Particle) -> str:
    parity = "-" if particle.parity < 0 else "+"
    if particle.spin.denominator == 1:
        spin = sp.latex(particle.spin)
    else:
        spin = Rf"\frac{{{particle.spin.numerator}}}{{{particle.spin.denominator}}}"
    return f"{spin}^{parity}"


def as_markdown_table(obj: Sequence) -> str:
    """Render objects a `str` suitable for generating a table."""
    item_type = _determine_item_type(obj)
    if item_type is Particle:
        return _as_resonance_markdown_table(obj)
    if item_type is ThreeBodyDecayChain:
        return _as_decay_markdown_table(obj)
    raise NotImplementedError(
        f"Cannot render a sequence with {item_type.__name__} items as a Markdown table"
    )


def _determine_item_type(obj: Sequence) -> type:
    if len(obj) < 1:
        raise ValueError(f"Need at least one entry to render a table")
    item_type = type(obj[0])
    if not all(map(lambda i: isinstance(i, item_type), obj)):
        raise ValueError(f"Not all items are of type {item_type.__name__}")
    return item_type


def _as_resonance_markdown_table(items: Sequence[Particle]) -> str:
    have_lineshapes = any(map(lambda p: p.lineshape is not None, items))
    column_names = [
        "name",
        "LaTeX",
        "$J^P$",
        "mass (MeV)",
        "width (MeV)",
    ]
    if have_lineshapes:
        column_names.append("lineshape")
    src = _create_markdown_table_header(column_names)
    for particle in items:
        row_items = [
            particle.name,
            f"${particle.latex}$",
            Rf"${as_latex(particle, only_jp=True)}$",
            f"{int(1e3 * particle.mass):,.0f}",
            f"{int(1e3 * particle.width):,.0f}",
        ]
        if have_lineshapes:
            row_items.append(particle.lineshape)
        src += _create_markdown_table_row(row_items)
    return src


def _as_decay_markdown_table(decays: Sequence[ThreeBodyDecayChain]) -> str:
    column_names = [
        "resonance",
        R"$J^P$",
        R"mass (MeV)",
        R"width (MeV)",
        R"$L_\mathrm{dec}^\mathrm{min}$",
        R"$L_\mathrm{prod}^\mathrm{min}$",
        "lineshape",
    ]
    src = _create_markdown_table_header(column_names)
    for decay in decays:
        child1, child2 = map(as_latex, decay.decay_products)
        row_items = [
            Rf"${decay.resonance.latex} \to" Rf" {child1} {child2}$",
            Rf"${as_latex(decay.resonance, only_jp=True)}$",
            f"{int(1e3 * decay.resonance.mass):,.0f}",
            f"{int(1e3 * decay.resonance.width):,.0f}",
            decay.outgoing_ls.L,
            decay.incoming_ls.L,
            decay.resonance.lineshape,
        ]
        src += _create_markdown_table_row(row_items)
    return src


def _create_markdown_table_header(column_names: list[str]):
    src = _create_markdown_table_row(column_names)
    src += _create_markdown_table_row(["---" for _ in column_names])
    return src


def _create_markdown_table_row(items: Iterable):
    items = map(lambda i: f"{i}", items)
    return "| " + " | ".join(items) + " |\n"


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

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

import hashlib
import json
import logging
import os
import pickle
from collections import abc
from functools import lru_cache, singledispatch
from os.path import abspath, dirname, expanduser
from textwrap import dedent
from typing import Iterable, Mapping, Sequence

import jax.numpy as jnp
import sympy as sp
from ampform.sympy import UnevaluatedExpression
from IPython.display import Math, display

from polarization.decay import IsobarNode, Particle, ThreeBodyDecay, ThreeBodyDecayChain

_LOGGER = logging.getLogger(__name__)


@singledispatch
def as_latex(obj, **kwargs) -> str:
    """Render objects as a LaTeX `str`.

    The resulting `str` can for instance be given to `IPython.display.Math`.

    Optional keywords:

    - only_jp: Render a `.Particle` as :math:`J^P` value (spin-parity) only.
    - with_jp: Render a `.Particle` with value :math:`J^P` value.
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
    return as_latex(obj.chains, **kwargs)


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
    if item_type is ThreeBodyDecay:
        return _as_decay_markdown_table(obj.chains)
    if item_type is ThreeBodyDecayChain:
        return _as_decay_markdown_table(obj)
    raise NotImplementedError(
        f"Cannot render a sequence with {item_type.__name__} items as a Markdown table"
    )


def _determine_item_type(obj) -> type:
    if not isinstance(obj, abc.Sequence):
        return type(obj)
    if len(obj) < 1:
        raise ValueError(f"Need at least one entry to render a table")
    item_type = type(obj[0])
    if not all(map(lambda i: isinstance(i, item_type), obj)):
        raise ValueError(f"Not all items are of type {item_type.__name__}")
    return item_type


def _as_resonance_markdown_table(items: Sequence[Particle]) -> str:
    column_names = [
        "name",
        "LaTeX",
        "$J^P$",
        "mass (MeV)",
        "width (MeV)",
    ]
    src = _create_markdown_table_header(column_names)
    for particle in items:
        row_items = [
            particle.name,
            f"${particle.latex}$",
            Rf"${as_latex(particle, only_jp=True)}$",
            f"{int(1e3 * particle.mass):,.0f}",
            f"{int(1e3 * particle.width):,.0f}",
        ]
        src += _create_markdown_table_row(row_items)
    return src


def _as_decay_markdown_table(decay_chains: Sequence[ThreeBodyDecayChain]) -> str:
    column_names = [
        "resonance",
        R"$J^P$",
        R"mass (MeV)",
        R"width (MeV)",
        R"$L_\mathrm{dec}^\mathrm{min}$",
        R"$L_\mathrm{prod}^\mathrm{min}$",
    ]
    src = _create_markdown_table_header(column_names)
    for chain in decay_chains:
        child1, child2 = map(as_latex, chain.decay_products)
        row_items = [
            Rf"${chain.resonance.latex} \to" Rf" {child1} {child2}$",
            Rf"${as_latex(chain.resonance, only_jp=True)}$",
            f"{int(1e3 * chain.resonance.mass):,.0f}",
            f"{int(1e3 * chain.resonance.width):,.0f}",
            chain.outgoing_ls.L,
            chain.incoming_ls.L,
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


def perform_cached_doit(
    unevaluated_expr: sp.Expr, directory: str | None = None
) -> sp.Expr:
    """Perform :code:`doit()` on an `~sympy.core.expr.Expr` and cache the result to disk.

    The cached result is fetched from disk if the hash of the original expression is the
    same as the hash embedded in the filename.

    Args:
        unevaluated_expr: A `sympy.Expr <sympy.core.expr.Expr>` on which to call
            :code:`doit()`.
        directory: The directory in which to cache the result. If `None`, the cache
            directory will be put under the source code directory where `polarization`
            is installed.

    .. tip:: For a faster cache, set `PYTHONHASHSEED
        <https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED>`_ to a
        fixed value.
    """
    if directory is None:
        home_directory = expanduser("~")
        directory = abspath(f"{home_directory}/.sympy-cache")
    h = get_readable_hash(unevaluated_expr)
    filename = f"{directory}/{h}.pkl"
    os.makedirs(dirname(filename), exist_ok=True)
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    _LOGGER.info(f"Cached expression file {filename} not found, performing doit()...")
    unfolded_expr = unevaluated_expr.doit()
    with open(filename, "wb") as f:
        pickle.dump(unfolded_expr, f)
    return unfolded_expr


def get_readable_hash(obj) -> str:
    b = _to_bytes(obj)
    return hashlib.sha256(b).hexdigest()


def _to_bytes(obj) -> bytes:
    python_hash_seed = os.environ.get("PYTHONHASHSEED", "")
    if python_hash_seed is not None and python_hash_seed.isdigit():
        # https://github.com/sympy/sympy/issues/14835#issuecomment-399782969
        python_hash_seed = int(python_hash_seed)
        return pickle.dumps(obj)
    if isinstance(obj, sp.Expr):
        # Using the str printer is slower and not necessarily unique,
        # but pickle.dumps() does not always result in the same bytes stream.
        _warn_about_unsafe_hash()
        return str(obj).encode()
    return pickle.dumps(obj)


@lru_cache(maxsize=None)  # warn once
def _warn_about_unsafe_hash():
    message = """
    PYTHONHASHSEED has not been set. For safer hashing of SymPy expressions, set the
    PYTHONHASHSEED environment variable to a fixed value and rerun the program. See
    https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED
    """
    message = dedent(message).replace("\n", " ").strip()
    _LOGGER.warning(message)


def mute_jax_warnings() -> None:
    jax_logger = logging.getLogger("absl")
    jax_logger.setLevel(logging.ERROR)


def export_polarization_field(
    sigma1: jnp.ndarray,
    sigma2: jnp.ndarray,
    alpha_x: jnp.ndarray,
    alpha_y: jnp.ndarray,
    alpha_z: jnp.ndarray,
    intensity: jnp.ndarray,
    filename: str,
) -> None:
    if len(sigma1.shape) != 1:
        raise ValueError(f"sigma1 must be a 1D array, got {sigma1.shape}")
    if len(sigma2.shape) != 1:
        raise ValueError(f"sigma2 must be a 1D array, got {sigma2.shape}")
    expected_shape: tuple[int, int] = (*sigma1.shape, *sigma2.shape)
    for array in [alpha_x, alpha_y, alpha_z, intensity]:
        if array.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {array.shape}")
    json_data = {
        "m^2_Kpi": sigma1.tolist(),
        "m^2_pK": sigma2.tolist(),
        "alpha_x": alpha_x.tolist(),
        "alpha_y": alpha_y.tolist(),
        "alpha_z": alpha_z.tolist(),
        "intensity": intensity.tolist(),
    }
    with open(filename, "w") as f:
        json.dump(json_data, f, separators=(",", ":"))


def import_polarization_field(filename: str, steps: int = 1) -> dict[str, jnp.ndarray]:
    with open(filename) as f:
        json_data: dict = json.load(f)
    return {
        "m^2_Kpi": jnp.array(json_data["m^2_Kpi"])[::steps],
        "m^2_pK": jnp.array(json_data["m^2_pK"])[::steps],
        "alpha_x": jnp.array(json_data["alpha_x"])[::steps, ::steps],
        "alpha_y": jnp.array(json_data["alpha_y"])[::steps, ::steps],
        "alpha_z": jnp.array(json_data["alpha_z"])[::steps, ::steps],
        "intensity": jnp.array(json_data["intensity"])[::steps, ::steps],
    }

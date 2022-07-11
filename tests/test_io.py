# pyright: reportPrivateUsage=false
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import pytest
import sympy as sp

from polarization.decay import IsobarNode, Particle
from polarization.io import _warn_about_unsafe_hash, as_latex, get_readable_hash

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture


# https://compwa-org--129.org.readthedocs.build/report/018.html#resonances-and-ls-scheme
dummy_args = dict(mass=0, width=0)
Λc = Particle("Λc", latex=R"\Lambda_c^+", spin=0.5, parity=+1, **dummy_args)
p = Particle("p", latex="p", spin=0.5, parity=+1, **dummy_args)
π = Particle("π+", latex=R"\pi^+", spin=0, parity=-1, **dummy_args)
K = Particle("K-", latex="K^-", spin=0, parity=-1, **dummy_args)
Λ1520 = Particle("Λ(1520)", latex=R"\Lambda(1520)", spin=1.5, parity=-1, **dummy_args)


def test_as_latex_particle():
    latex = as_latex(Λ1520)
    assert latex == Λ1520.latex
    latex = as_latex(Λ1520, only_jp=True)
    assert latex == R"\frac{3}{2}^-"
    latex = as_latex(Λ1520, with_jp=True)
    assert latex == Λ1520.latex + R"\left[\frac{3}{2}^-\right]"


def test_as_latex_isobar_node():
    node = IsobarNode(Λ1520, p, K)
    latex = as_latex(node)
    assert latex == R"\Lambda(1520) \to p K^-"
    latex = as_latex(node, with_jp=True)
    assert (
        latex == R"\Lambda(1520)\left[\frac{3}{2}^-\right] \to"
        R" p\left[\frac{1}{2}^+\right] K^-\left[0^-\right]"
    )

    node = IsobarNode(Λ1520, p, K, interaction=(2, 1))
    latex = as_latex(node)
    assert latex == R"\Lambda(1520) \xrightarrow[S=1]{L=2} p K^-"


@pytest.mark.parametrize(
    ("assumptions", "expected_hash"),
    [
        (dict(), "34702c2"),
        (dict(real=True), "440b870"),
        (dict(rational=True), "f308f4c"),
    ],
)
def test_get_readable_hash(assumptions, expected_hash, caplog: LogCaptureFixture):
    caplog.set_level(logging.WARNING)
    x, y = sp.symbols("x y", **assumptions)
    expr = x**2 + y
    h = get_readable_hash(expr)
    python_hash_seed = os.environ.get("PYTHONHASHSEED")
    if python_hash_seed is None or not python_hash_seed.isdigit():
        assert h[:7] == "bbc9833"
        if _warn_about_unsafe_hash.cache_info().hits == 0:
            assert "PYTHONHASHSEED has not been set." in caplog.text
            caplog.clear()
    elif python_hash_seed == "0":
        assert h[:7] == expected_hash
    else:
        pytest.skip("PYTHONHASHSEED has been set, but is not 0")
    assert caplog.text == ""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import logging
import os
from os.path import abspath, dirname
from typing import TYPE_CHECKING

import pytest
import sympy as sp

from polarimetry import formulate_polarimetry
from polarimetry.decay import IsobarNode, Particle
from polarimetry.io import _warn_about_unsafe_hash, as_latex, get_readable_hash
from polarimetry.lhcb import load_model_builder
from polarimetry.lhcb.particle import load_particles

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture

    from polarimetry.amplitude import DalitzPlotDecompositionBuilder

THIS_DIR = dirname(abspath(__file__))

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
        (dict(), "pythonhashseed-0+7459658071388516764"),
        (dict(real=True), "pythonhashseed-0+3665410414623666716"),
        (dict(rational=True), "pythonhashseed-0-7926839224244779605"),
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
        assert h == expected_hash
    else:
        pytest.skip("PYTHONHASHSEED has been set, but is not 0")
    assert caplog.text == ""


@pytest.mark.slow()
@pytest.mark.parametrize(
    ("model_id", "intensity_hash", "polarimetry_hash"),
    [
        (0, 72624733, 50993229),
        (1, 72624733, 50993229),
        (2, 72624733, 50993229),
        (3, 72624733, 50993229),
        (4, 72624733, 50993229),
        (5, 72624733, 50993229),
        (6, 72624733, 50993229),
        (7, 42667828, 59774787),
        (8, 42079337, 75917912),
        (9, 49930455, 66970314),
        (10, 62675824, 87880477),
        (11, 72624733, 50993229),
        (12, 57901794, 70431636),
        (13, 89118263, 27958629),
        (14, 72624733, 50993229),
        (15, 55246840, 89733841),
        (16, 72624733, 50993229),
        (17, 96889798, 22186476),
    ],
)
def test_get_readable_hash_large(model_id, intensity_hash, polarimetry_hash):
    python_hash_seed = os.environ.get("PYTHONHASHSEED")
    if python_hash_seed != "0":
        pytest.skip("PYTHONHASHSEED is not 0")

    builder = __get_model_builder(model_id)
    model = builder.formulate(reference_subsystem=1)
    h = get_readable_hash(model.full_expression)
    short_hash = int(h[17:25])
    assert short_hash == intensity_hash
    polarimetry_exprs = formulate_polarimetry(builder, reference_subsystem=1)
    h = get_readable_hash(polarimetry_exprs[0].doit().xreplace(model.amplitudes))
    short_hash = int(h[17:25])
    assert short_hash == polarimetry_hash


def __get_model_builder(model_id: int | str) -> DalitzPlotDecompositionBuilder:
    data_dir = f"{THIS_DIR}/../data"
    model_file = f"{data_dir}/model-definitions.yaml"
    particles = load_particles(f"{data_dir}/particle-definitions.yaml")
    return load_model_builder(model_file, particles, model_id)

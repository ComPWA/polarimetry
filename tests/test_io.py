# pyright: reportPrivateUsage=false
from __future__ import annotations

import logging
import os
from os.path import abspath, dirname
from typing import TYPE_CHECKING

import pytest
import sympy as sp

from polarization import formulate_polarization
from polarization.amplitude import DalitzPlotDecompositionBuilder
from polarization.decay import IsobarNode, Particle
from polarization.io import _warn_about_unsafe_hash, as_latex, get_readable_hash
from polarization.lhcb import load_model_builder
from polarization.lhcb.particle import load_particles

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture

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


@pytest.mark.parametrize(
    ("model_id", "intensity_hash", "polarization_hash"),
    [
        (0, "e6713ea", "ce83bd4"),
        (1, "e6713ea", "d220bcf"),
        (2, "e6713ea", "d220bcf"),
        (3, "e6713ea", "d220bcf"),
        (4, "e6713ea", "d220bcf"),
        (5, "e6713ea", "d220bcf"),
        (6, "e6713ea", "d220bcf"),
        (7, "d283d6e", "7988df7"),
        (8, "ba44b94", "678ca1b"),
        (9, "15c4be1", "119ffeb"),
        (10, "a6aaf57", "1381db4"),
        (11, "e6713ea", "d220bcf"),
        (12, "7c9726b", "411e9ba"),
        (13, "2e8122c", "51e325e"),
        (14, "e6713ea", "d220bcf"),
        # 15: No dynamics implemented for lineshape "BuggBreitWignerExpFF"
        (16, "e6713ea", "d220bcf"),
        # 17: No dynamics implemented for lineshape "Flatte1405_LS"
    ],
)
def test_get_readable_hash_large(model_id, intensity_hash, polarization_hash):
    python_hash_seed = os.environ.get("PYTHONHASHSEED")
    if python_hash_seed != "0":
        pytest.skip("PYTHONHASHSEED is not 0")

    builder = __get_model_builder(model_id)
    model = builder.formulate(reference_subsystem=1)
    h = get_readable_hash(model.full_expression)
    assert h[:7] == intensity_hash
    polarization_exprs = formulate_polarization(builder, reference_subsystem=1)
    h = get_readable_hash(polarization_exprs[0].doit().xreplace(model.amplitudes))
    assert h[:7] == polarization_hash


def __get_model_builder(model_id: int | str) -> DalitzPlotDecompositionBuilder:
    data_dir = f"{THIS_DIR}/../data"
    model_file = f"{data_dir}/model-definitions.yaml"
    particles = load_particles(f"{data_dir}/particle-definitions.yaml")
    return load_model_builder(model_file, particles, model_id)

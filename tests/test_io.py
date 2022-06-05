from pathlib import Path

import pytest
import sympy as sp

from polarization.decay import IsobarNode, Particle
from polarization.io import as_latex, get_readable_hash
from polarization.lhcb import load_resonance_definitions

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
    "assumptions",
    [
        dict(),
        dict(real=True),
        dict(rational=True),
    ],
)
def test_get_readable_hash(assumptions):
    x, y = sp.symbols("x y", **assumptions)
    expr = x**2 + y
    h = get_readable_hash(expr)
    assert h == "bbc98339949be8bbeb405eb320f2b42d24c597cf0a8780408070d28a320d16fc"
    # Assumptions do not affect the hash. This should be addressed through:
    # https://github.com/redeboer/polarization-sensitivity/issues/41


def test_load_isobar_definitions():
    pwd = Path(__file__).absolute().parent
    resonances = load_resonance_definitions(pwd.parent / "data" / "isobars.json")
    assert len(resonances) == 12
    Λ2000 = resonances["L(2000)"]
    assert Λ2000.name == "L(2000)"
    assert Λ2000.latex == "L(2000)"
    assert Λ2000.spin == 0.5
    assert Λ2000.parity == -1
    assert Λ2000.mass == 2.0
    assert Λ2000.width == 0.21
    assert Λ2000.lineshape == "BreitWignerMinL"

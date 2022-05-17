from pathlib import Path

from polarization.decay import IsobarNode, Particle
from polarization.io import as_latex
from polarization.lhcb import load_resonance_definitions

# https://compwa-org--129.org.readthedocs.build/report/018.html#resonances-and-ls-scheme
Λc = Particle("Λc", latex=R"\Lambda_c^+", spin=0.5, parity=+1)
p = Particle("p", latex="p", spin=0.5, parity=+1)
π = Particle("π+", latex=R"\pi^+", spin=0, parity=-1)
K = Particle("K-", latex="K^-", spin=0, parity=-1)
Λ1520 = Particle("Λ(1520)", latex=R"\Lambda(1520)", spin=1.5, parity=-1)


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


def test_load_isobar_definitions():
    pwd = Path(__file__).absolute().parent
    resonances = load_resonance_definitions(pwd.parent / "data" / "isobars.json")
    assert len(resonances) == 12
    Λ2000 = resonances["L(2000)"]
    assert Λ2000.name == "L(2000)"
    assert Λ2000.latex == "L(2000)"
    assert Λ2000.spin == 0.5
    assert Λ2000.parity == -1
    assert Λ2000.mass_range == (1900, 2100)
    assert Λ2000.mass == 2000
    assert Λ2000.width_range == (20, 400)
    assert Λ2000.width == 210
    assert Λ2000.lineshape == "BreitWignerMinL"

import json
from pathlib import Path

from polarization.decay import IsobarNode, Particle
from polarization.io import as_latex, to_resonance_dict

# https://compwa-org--129.org.readthedocs.build/report/018.html#resonances-and-ls-scheme
Λc = Particle(R"\Lambda_c^+", spin=0.5, parity=+1)
p = Particle("p", spin=0.5, parity=+1)
π = Particle(R"\pi^+", spin=0, parity=-1)
K = Particle("K^-", spin=0, parity=-1)
Λ1520 = Particle(R"\Lambda(1520)", spin=1.5, parity=-1)


def test_as_latex_particle():
    latex = as_latex(Λ1520)
    assert latex == Λ1520.name
    latex = as_latex(Λ1520, render_jp=True)
    assert latex == R"{3/2}^{-1}"


def test_as_latex_isobar_node():
    node = IsobarNode(Λ1520, p, K)
    latex = as_latex(node)
    assert latex == R"\Lambda(1520) \to p K^-"
    latex = as_latex(node, render_jp=True)
    assert latex == R"{3/2}^{-1} \to {1/2}^{+1} {0}^{-1}"

    node = IsobarNode(Λ1520, p, K, interaction=(2, 1))
    latex = as_latex(node)
    assert latex == R"\Lambda(1520) \xrightarrow[S=1]{L=2} p K^-"


def test_import_isobar_definitions():
    pwd = Path(__file__).absolute().parent
    data_dir = pwd.parent / "data"
    with open(data_dir / "isobars.json") as stream:
        data = json.load(stream)
    isobar_definitions = data["isobars"]
    resonances = to_resonance_dict(isobar_definitions)
    assert len(resonances) == 12
    Λ2000 = resonances["L(2000)"]
    assert Λ2000.particle.name == "L(2000)"
    assert Λ2000.particle.spin == 0.5
    assert Λ2000.particle.parity == -1
    assert Λ2000.mass_range == (1900, 2100)
    assert Λ2000.mass == 2000
    assert Λ2000.width_range == (20, 400)
    assert Λ2000.width == 210
    assert Λ2000.lineshape == "BreitWignerMinL"

from polarization.decay import IsobarNode, Particle
from polarization.io import as_latex

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

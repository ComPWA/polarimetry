from polarization.decay import IsobarNode, Particle

# https://compwa-org--129.org.readthedocs.build/report/018.html#resonances-and-ls-scheme
Λc = Particle(R"\Lambda_c^+", spin=0.5, parity=+1)
p = Particle("p", spin=0.5, parity=+1)
π = Particle(R"\pi^+", spin=0, parity=-1)
K = Particle("K^-", spin=0, parity=-1)
Λ1520 = Particle(R"\Lambda(1520)", spin=1.5, parity=-1)


class TestIsobarNode:
    def test_children(self):
        decay = IsobarNode(Λ1520, p, K)
        assert decay.children == (p, K)

    def test_ls(self):
        L, S = 2, 1
        node = IsobarNode(Λ1520, p, K, interaction=(L, S))
        assert node.interaction is not None
        assert node.interaction.L == L
        assert node.interaction.S == S

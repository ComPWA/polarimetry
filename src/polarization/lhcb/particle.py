"""Hard-coded particle definitions."""
from polarization.decay import Particle

Λc = Particle(
    name="Λc⁺",
    latex=R"\Lambda_c^+",
    spin=0.5,
    parity=+1,
    mass=2.28646,
    width=3.25e-12,
)
p = Particle(
    name="p",
    latex="p",
    spin=0.5,
    parity=+1,
    mass=0.938272046,
    width=0.0,
)
K = Particle(
    name="K⁻",
    latex="K^-",
    spin=0,
    parity=-1,
    mass=0.493677,
    width=5.317e-17,
)
π = Particle(
    name="π⁺",
    latex=R"\pi^+",
    spin=0,
    parity=-1,
    mass=0.13957018,
    width=2.5284e-17,
)
PARTICLE_TO_ID = {Λc: 0, p: 1, π: 2, K: 3}

# https://github.com/redeboer/polarization-sensitivity/blob/34f5330/julia/notebooks/model0.jl#L43-L47
Σ = Particle(
    name="Σ⁻",
    latex=R"\Sigma^-",
    spin=0.5,
    parity=+1,
    mass=1.18937,
    width=4.45e-15,
)

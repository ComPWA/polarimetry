import sympy as sp
from ampform.kinematics.phasespace import Kallen

from polarization.amplitude import formulate_scattering_angle

m0, m1, m2, m3 = sp.symbols("m:4", nonnegative=True)
σ1, σ2, σ3 = sp.symbols("sigma1:4", nonnegative=True)


def test_formulate_scattering_angle():
    assert formulate_scattering_angle(2, 3) == sp.acos(
        (
            2 * σ1 * (-(m1**2) - m2**2 + σ3)
            - (m0**2 - m1**2 - σ1) * (m2**2 - m3**2 + σ1)
        )
        / (
            sp.sqrt(Kallen(m0**2, m1**2, σ1))
            * sp.sqrt(Kallen(σ1, m2**2, m3**2))
        )
    )
    assert formulate_scattering_angle(3, 1) == sp.acos(
        (
            2 * σ2 * (-(m2**2) - m3**2 + σ1)
            - (m0**2 - m2**2 - σ2) * (-(m1**2) + m3**2 + σ2)
        )
        / (
            sp.sqrt(Kallen(m0**2, m2**2, σ2))
            * sp.sqrt(Kallen(σ2, m3**2, m1**2))
        )
    )

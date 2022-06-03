from __future__ import annotations

import sympy as sp

from polarization.dynamics import Kallen


def formulate_scattering_angle(
    state_id: int, sibling_id: int
) -> tuple[sp.Symbol, sp.acos]:
    r"""Formulate the scattering angle in the rest frame of the resonance.

    Compute the :math:`\theta_{ij}` scattering angle as formulated in `Eq (A1) in the
    DPD paper <https://journals.aps.org/prd/pdf/10.1103/PhysRevD.101.034033#page=9>`_.
    The angle is that between particle :math:`i` and spectator particle :math:`k` in the
    rest frame of the isobar resonance :math:`(ij)`.
    """
    if not {state_id, sibling_id} <= {1, 2, 3}:
        raise ValueError(f"Child IDs need to be one of 1, 2, 3")
    if state_id == sibling_id:
        raise ValueError(f"IDs of the decay products cannot be equal: {state_id}")
    symbol = sp.Symbol(Rf"theta_{state_id}{sibling_id})", real=True)
    spectator_id = next(iter({1, 2, 3} - {state_id, sibling_id}))
    m0 = sp.Symbol(f"m0", nonnegative=True)
    mi = sp.Symbol(f"m{state_id}", nonnegative=True)
    mj = sp.Symbol(f"m{sibling_id}", nonnegative=True)
    mk = sp.Symbol(f"m{spectator_id}", nonnegative=True)
    σj = sp.Symbol(f"sigma{sibling_id}", nonnegative=True)
    σk = sp.Symbol(f"sigma{spectator_id}", nonnegative=True)
    theta = sp.acos(
        (
            2 * σk * (σj - mk**2 - mi**2)
            - (σk + mi**2 - mj**2) * (m0**2 - σk - mk**2)
        )
        / (
            sp.sqrt(Kallen(m0**2, mk**2, σk))
            * sp.sqrt(Kallen(σk, mi**2, mj**2))
        )
    )
    return symbol, theta

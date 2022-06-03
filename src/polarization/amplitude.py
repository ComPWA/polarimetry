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


def formulate_theta_hat_angle(
    isobar_id: int, alignment_system: int
) -> tuple[sp.Symbol, sp.acos]:
    r"""Formulate an expression for :math:`\hat\theta_{i(j)}`."""
    allowed_ids = {1, 2, 3}
    if not {isobar_id, alignment_system} <= allowed_ids:
        raise ValueError(f"Child IDs need to be one of {', '.join(allowed_ids)}")
    symbol = sp.Symbol(Rf"theta_{isobar_id}({alignment_system})", real=True)
    if isobar_id == alignment_system:
        return symbol, sp.S.Zero
    if (isobar_id, alignment_system) in {(3, 1), (1, 2), (2, 3)}:
        remaining_id = next(iter(allowed_ids - {isobar_id, alignment_system}))
        m0 = sp.Symbol(f"m0", nonnegative=True)
        mi = sp.Symbol(f"m{isobar_id}", nonnegative=True)
        mj = sp.Symbol(f"m{alignment_system}", nonnegative=True)
        σi = sp.Symbol(f"sigma{isobar_id}", nonnegative=True)
        σj = sp.Symbol(f"sigma{alignment_system}", nonnegative=True)
        σk = sp.Symbol(f"sigma{remaining_id}", nonnegative=True)
        theta = sp.acos(
            (
                (m0**2 + mi**2 - σi) * (m0**2 + mj**2 - σj)
                - 2 * m0**2 * (σk - mi**2 - mj**2)
            )
            / (
                sp.sqrt(Kallen(m0**2, mj**2, σj))
                * sp.sqrt(Kallen(m0**2, σi, mi**2))
            )
        )
        return symbol, theta
    _, theta = formulate_theta_hat_angle(alignment_system, isobar_id)
    return symbol, -theta

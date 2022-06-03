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
        raise ValueError(
            f"Child IDs need to be one of {', '.join(map(str, allowed_ids))}"
        )
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


def formulate_zeta_angle(
    boosted_state: int,
    alignment_system: int,
    helicity_system: int,
) -> tuple[sp.Symbol, sp.acos]:
    r"""Formulate an expression for the alignment angle :math:`\zeta^i_{j(k)}`.

    Arguments
    ---------
    boosted_state: The ID of the state that is being boosted.
    alignment_system: The ID of the system that is being aligned.
    helicity_system: The ID of the system that is being rotated.
    """
    zeta_symbol = sp.Symbol(
        Rf"\cos(\zeta^{boosted_state}_{{{alignment_system}({helicity_system})}})",
        real=True,
    )
    if boosted_state == 0:
        _, theta = formulate_theta_hat_angle(alignment_system, helicity_system)
        return zeta_symbol, theta
    if helicity_system == 0:
        _, zeta = formulate_zeta_angle(boosted_state, alignment_system, boosted_state)
        return zeta_symbol, zeta
    if alignment_system == helicity_system:
        return zeta_symbol, sp.S.Zero
    m0, m1, m2, m3 = sp.symbols("m:4", nonnegative=True)
    σ1, σ2, σ3 = sp.symbols("sigma1:4", nonnegative=True)
    if (boosted_state, alignment_system, helicity_system) == (1, 1, 3):
        cos_zeta_expr = (
            2 * m1**2 * (σ2 - m0**2 - m2**2)
            + (m0**2 + m1**2 - σ1) * (σ3 - m1**2 - m2**2)
        ) / (
            sp.sqrt(Kallen(m0**2, m1**2, σ1))
            * sp.sqrt(Kallen(σ3, m1**2, m2**2))
        )
        return zeta_symbol, sp.acos(cos_zeta_expr)
    if (boosted_state, alignment_system, helicity_system) == (1, 2, 1):
        cos_zeta_expr = (
            2 * m1**2 * (σ3 - m0**2 - m3**2)
            + (m0**2 + m1**2 - σ1) * (σ2 - m1**2 - m3**2)
        ) / (
            sp.sqrt(Kallen(m0**2, m1**2, σ1))
            * sp.sqrt(Kallen(σ2, m1**2, m3**2))
        )
        return zeta_symbol, sp.acos(cos_zeta_expr)
    if (boosted_state, alignment_system, helicity_system) == (2, 2, 1):
        cos_zeta_expr = (
            2 * m2**2 * (σ3 - m0**2 - m3**2)
            + (m0**2 + m2**2 - σ2) * (σ1 - m2**2 - m3**2)
        ) / (
            sp.sqrt(Kallen(m0**2, m2**2, σ2))
            * sp.sqrt(Kallen(σ1, m2**2, m3**2))
        )
        return zeta_symbol, sp.acos(cos_zeta_expr)
    if (boosted_state, alignment_system, helicity_system) == (2, 3, 2):
        cos_zeta_expr = (
            2 * m2**2 * (σ1 - m0**2 - m1**2)
            + (m0**2 + m2**2 - σ2) * (σ3 - m2**2 - m1**2)
        ) / (
            sp.sqrt(Kallen(m0**2, m2**2, σ2))
            * sp.sqrt(Kallen(σ3, m2**2, m1**2))
        )
        return zeta_symbol, sp.acos(cos_zeta_expr)
    if (boosted_state, alignment_system, helicity_system) == (3, 3, 2):
        cos_zeta_expr = (
            2 * m3**2 * (σ1 - m0**2 - m1**2)
            + (m0**2 + m3**2 - σ3) * (σ2 - m3**2 - m1**2)
        ) / (
            sp.sqrt(Kallen(m0**2, m3**2, σ3))
            * sp.sqrt(Kallen(σ2, m3**2, m1**2))
        )
        return zeta_symbol, sp.acos(cos_zeta_expr)
    if (boosted_state, alignment_system, helicity_system) == (3, 1, 3):
        cos_zeta_expr = (
            2 * m3**2 * (σ2 - m0**2 - m2**2)
            + (m0**2 + m3**2 - σ3) * (σ1 - m3**2 - m2**2)
        ) / (
            sp.sqrt(Kallen(m0**2, m3**2, σ3))
            * sp.sqrt(Kallen(σ1, m3**2, m2**2))
        )
        return zeta_symbol, sp.acos(cos_zeta_expr)
    if (boosted_state, alignment_system, helicity_system) in {  # Eq (A10)
        (1, 2, 3),
        (2, 3, 1),
        (3, 1, 2),
    }:
        create_symbols = lambda i: sp.symbols(f"m{i} sigma{i}", nonnegative=True)
        mi, σi = create_symbols(boosted_state)
        mj, σj = create_symbols(alignment_system)
        mk, σk = create_symbols(helicity_system)
        cos_zeta_expr = (
            2 * mi**2 * (mj**2 + mk**2 - σi)
            + (σj + mi**2 - mk**2) * (σk - mi**2 - mj**2)
        ) / (
            sp.sqrt(Kallen(σj, mk**2, mi**2))
            * sp.sqrt(Kallen(σk, mi**2, mj**2))
        )
        return zeta_symbol, sp.acos(cos_zeta_expr)
    if (boosted_state, alignment_system, helicity_system) in {
        (1, 3, 1),
        (2, 1, 2),
        (3, 2, 3),
        # Eq (A8)
        (1, 1, 2),
        (2, 2, 3),
        (3, 3, 1),
        # Eq (A11)
        (1, 3, 2),
        (2, 1, 3),
        (3, 2, 1),
    }:
        _, zeta = formulate_zeta_angle(boosted_state, helicity_system, alignment_system)
        return zeta_symbol, -zeta
    raise NotImplementedError(
        f"No expression for ζ^{boosted_state}_{alignment_system}({helicity_system})"
    )

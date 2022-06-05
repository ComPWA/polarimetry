import sympy as sp
from ampform.sympy import PoolSum
from sympy.physics.matrices import msigma

from .amplitude import DalitzPlotDecompositionBuilder


def formulate_polarization(builder: DalitzPlotDecompositionBuilder):
    spins = [builder.decay.states[i].spin for i in builder.decay.states]
    half = sp.Rational(1, 2)
    if spins != [half, half, 0, 0]:
        raise ValueError(
            "Can only formulate polarization for an initial state with spin 1/2 and a"
            f" final state with spin 1/2, 0, 0, but got spins {spins}"
        )
    model = builder.formulate()
    λ_p, λ_Λc, λ_Λc_prime = sp.symbols(R"lambda nu \nu^{\prime}")
    return [
        PoolSum(
            builder.formulate_aligned_amplitude(λ_Λc, λ_p, 0, 0)[0].conjugate()
            * pauli_matrix[_to_index(λ_Λc), _to_index(λ_Λc_prime)]
            * builder.formulate_aligned_amplitude(λ_Λc_prime, λ_p, 0, 0)[0],
            (λ_Λc, [-half, +half]),
            (λ_Λc_prime, [-half, +half]),
            (λ_p, [-half, +half]),
        )
        / model.intensity
        for pauli_matrix in map(msigma, [1, 2, 3])
    ]


def _to_index(helicity):
    """Symbolic conversion of half-value helicities to Pauli matrix indices."""
    return sp.Piecewise(
        (1, sp.LessThan(helicity, 0)),
        (0, True),
    )

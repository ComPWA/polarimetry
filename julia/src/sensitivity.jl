two_Δλ(HRk) = HRk.two_λa - HRk.two_λb

const σPauli = [
    [0 1; 1 0],
    [0 -1im; 1im 0],
    [1 0; 0 -1],
    [1 0; 0 1]
]
#
twoλ2ind(twoλ) = 2 - div(twoλ + 1, 2)
#
expectation(Op, cs, Hs) = sum(
    conj(ci) * cj * Op[twoλ2ind(two_Δλ(Hi)), twoλ2ind(two_Δλ(Hj))] *
    (Hi.two_λb == Hj.two_λb) * (Hi.two_λa == Hj.two_λa)
    for (ci, Hi) in zip(cs, Hs),
    (cj, Hj) in zip(cs, Hs)) |> real
#

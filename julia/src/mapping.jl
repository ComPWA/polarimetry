
function selectindexmap(isobarname)
    #
    couplingindexmap = Dict(
        r"[L|D].*" => Dict(
            '1' => (1, 0),
            '2' => (-1, 0)),
        r"K\(892\)" => Dict(
            '1' => (0, 1),
            '2' => (-2, 1),
            '3' => (2, -1),
            '4' => (0, -1)),
        r"K\(700|1430\)" => Dict(
            '1' => (0, -1),
            '2' => (0, 1)))
    #
    m = filter(keys(couplingindexmap)) do k
        match(k, isobarname) !== nothing
    end
    return couplingindexmap[first(m)]
end

function parname2decaychain(parname, isobars)
    isobarname = parname[3:end-1]
    #
    @unpack k, Hij, two_s, Xlineshape, parity = isobars[isobarname]
    #
    couplingindex = parname[end]
    two_λR, two_λk = selectindexmap(isobarname)[couplingindex]
    two_λR′, two_λk′, c′ =
        couplingLHCb2DPD(two_λR, two_λk; k, two_j=two_s, parity)
    HRk = NoRecoupling(two_λR′, two_λk′)
    (c′, DecayChain(; k, Xlineshape, Hij, HRk, two_s, tbs))
end



function couplingLHCb2DPD(two_λR, two_λk; k, parity, two_j)
    if k == 2
        @assert two_λk == 0
        c′ = -(2 * (parity == '+') - 1) / sqrt(two_j + 1)
        return (-two_λR, two_λk, c′)
    elseif k == 3
        c′ = -(2 * (parity == '+') - 1) * minusone()^(two_j // 2 - 1 // 2) / sqrt(two_j + 1)
        return (-two_λR, two_λk, c′)
    end
    k != 1 && error("cannot be!")
    c′ = 1.0 / sqrt(two_j + 1)
    return (two_λR, -two_λk, c′)
end

"""
The relation is
```math
A^{DPD}_{λ₀,λ₁} = (-1)^{½-λ₁} A^{LHCb}_{λ₀,-λ₁}
```
"""
amplitudeLHCb2DPD(A) =
    [A[1, 2] -A[1, 1]
        A[2, 2] -A[2, 1]]
#

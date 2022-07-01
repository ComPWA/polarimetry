
intensity(Ai::AbstractArray, ci::AbstractVector) = sum(abs2, sum(a .* ci) for a in Ai)


struct LHCbModel{N,T,L<:Number}
    chains::SVector{N,DecayChain{X,V1,V2,T} where {X,V1,V2}}
    couplings::SVector{N,L}
    isobarnames::SVector{N,String}
end

function LHCbModel(; chains, couplings, isobarnames)
    N = length(chains)
    N != length(couplings) && error("Length of couplings does not match the length of the chains")
    N != length(isobarnames) && error("Length of isobarnames does not match the length of the chains")
    #
    Ttbs = typeof(chains[1].tbs)
    sv_chains = (SVector{N,DecayChain{X,V1,V2,Ttbs} where {X,V1,V2}})(chains)
    sv_couplings = SVector{N}(couplings)
    sv_isobarnames = SVector{N}(isobarnames)
    #
    LHCbModel(sv_chains, sv_couplings, sv_isobarnames)
end


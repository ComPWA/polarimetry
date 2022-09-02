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
    v = collect(zip(chains, couplings, isobarnames))
    sort!(v, by=x -> eval(Meta.parse(x[3][3:end-1])))
    sort!(v, by=x -> findfirst(x[3][1], "LDK"))
    #
    sort_chains, sort_couplings, sort_isobarnames =
        getindex.(v, 1), getindex.(v, 2), getindex.(v, 3)
    #
    Ttbs = typeof(chains[1].tbs)
    sv_sort_chains = (SVector{N,DecayChain{X,V1,V2,Ttbs} where {X,V1,V2}})(sort_chains)
    sv_sort_couplings = SVector{N}(sort_couplings)
    sv_sort_isobarnames = SVector{N}(sort_isobarnames)
    #
    LHCbModel(sv_sort_chains, sv_sort_couplings, sv_sort_isobarnames)
end


amplitude(model::LHCbModel, σs, two_λs) =
    sum(c * amplitude(d, σs, two_λs) for (c, d) in zip(model.couplings, model.chains))

intensity(model::LHCbModel, σs) =
    sum(abs2, amplitude(model, σs, two_λs)
              for two_λs in itr(tbs.two_js))

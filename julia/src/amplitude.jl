
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



function LHCbModel(modeldict; particledict)

    # 1) get isobars
    isobars = Dict()
    for (key, lineshape) in modeldict["lineshapes"]
        dict = Dict{String,Any}(particledict[key])
        dict["lineshape"] = lineshape
        isobars[key] = buildchain(key, dict)
    end

    defaultparameters = modeldict["parameters"]
    shapeparameters = filter(x -> x[1] != 'A', keys(defaultparameters))
    parameterupdates = [
        replacementpair(parname, defaultparameters[parname])
        for parname in shapeparameters]

    for (p, u) in parameterupdates
        BW = isobars[p].Xlineshape
        isobars[p] = merge(isobars[p],
            (Xlineshape=updatepars(BW, merge(BW.pars, u)),))
    end

    # 3) get couplings
    couplingkeys = collect(filter(x -> x[1:2] == "Ar", keys(defaultparameters)))
    isobarnames = map(x -> x[3:end-1], couplingkeys)

    terms = []
    for parname in couplingkeys
        c_re_key = "Ar" * parname[3:end]
        c_im_key = "Ai" * parname[3:end]
        value_re = eval(Meta.parse(defaultparameters[c_re_key])).val
        value_im = eval(Meta.parse(defaultparameters[c_im_key])).val
        value = value_re + 1im * value_im
        #
        c0, d = parname2decaychain(parname, isobars)
        #
        push!(terms, (c0 * value, d))
    end

    chains = getindex.(terms, 2)
    couplings = getindex.(terms, 1)

    LHCbModel(; chains, couplings, isobarnames)
end

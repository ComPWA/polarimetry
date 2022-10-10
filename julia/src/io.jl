
function readjson(path)
    f = read(path, String)
    return JSON.parse(f)
end

function writejson(path, obj)
    open(path, "w") do io
        JSON.print(io, obj, 4)
    end
end

"""
    ifhyphenaverage(s::String)

The function treats the shape-parameter fields in the particle-description file.
If range is provided, it averages the limits.
"""
function ifhyphenaverage(s::String)
    factor = findfirst('-', s) === nothing ? 1 : 2
    eval(Meta.parse(replace(s, '-' => '+'))) / factor
end
ifhyphenaverage(v::Number) = v


function definechaininputs(key, dict)
    @unpack mass, width, lineshape = dict
    #
    k = Dict('K' => 1, 'D' => 3, 'L' => 2)[first(key)]
    #
    jp_R = str2jp(dict["jp"])
    parity = jp_R.p
    two_j = jp_R.j |> x2
    #
    massval, widthval = ifhyphenaverage.((mass, width)) ./ 1e3
    #
    i, j = ij_from_k(k)
    #
    @unpack two_js = tbs
    #
    reaction_ij = jp_R => (jp(two_js[i] // 2, parities[i]), jp(two_js[j] // 2, parities[j]))
    reaction_Rk(P0) = jp(two_js[4] // 2, P0) => (jp_R, jp(two_js[k] // 2, parities[k]))
    #
    LS = vcat(possible_ls.(reaction_Rk.(('+', '-')))...)
    minLS = first(sort(vcat(LS...); by=x -> x[1]))
    #
    ls = possible_ls(reaction_ij)
    length(ls) != 1 && error("expected the only ls: $(ls)")
    onlyls = first(ls)
    #
    Hij = ParityRecoupling(two_js[i], two_js[j], reaction_ij)
    Xlineshape = eval(
        quote
            $(Symbol(lineshape))(
                (; m=$massval, Γ=$widthval);
                name=$key,
                l=$(onlyls[1]),
                minL=$(minLS[1]),
                m1=$(ms[i]), m2=$(ms[j]), mk=$(ms[k]), m0=$(ms[4]))
        end)
    return (; k, Xlineshape, Hij, two_s=two_j, parity)
end



# shape parameters
function parseshapedparameter(parname)
    keytemp = r"([M,G]|gamma|alpha)"
    nametemp = r"([L,K,D]\([0-9]*\))"
    m = match(keytemp * nametemp, parname)
    return (key=m[1], isobarname=m[2])
end

function keyname2symbol(key)
    key == "M" && return :m
    key == "G" && return :Γ
    key == "gamma" && return :γ
    key == "alpha" && return :α
    error("The name of the shared parameter, $(key), is not recognized!")
end

function replacementpair(parname, val)
    @unpack key, isobarname = parseshapedparameter(parname)
    s = keyname2symbol(key)
    v = MeasuredParameter(val).val
    isobarname => eval(:(NamedTuple{($(QuoteNode(s)),)}($(v))))
end



function LHCbModel(modeldict; particledict)

    # 1) get isobars
    isobars = Dict()
    for (key, lineshape) in modeldict["lineshapes"]
        dict = Dict{String,Any}(particledict[key])
        dict["lineshape"] = lineshape
        isobars[key] = definechaininputs(key, dict)
    end

    # 3) get parameters
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
        value_re = MeasuredParameter(defaultparameters[c_re_key]).val
        value_im = MeasuredParameter(defaultparameters[c_im_key]).val
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

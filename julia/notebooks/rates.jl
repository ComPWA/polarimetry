### A Pluto.jl notebook ###
# v0.19.5

using Markdown
using InteractiveUtils

# ╔═╡ 03733bd2-dcf3-11ec-231f-8dab0ad6b19e
begin
    cd(joinpath(@__DIR__, ".."))
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
    #
    using JSON
    using Plots
    using LaTeXStrings
    import Plots.PlotMeasures.mm
    using RecipesBase
    #
    using LinearAlgebra
    using Parameters
    using Measurements
    using DataFrames
    using ThreadsX
    #
    using ThreeBodyDecay
    using ThreeBodyDecay.PartialWaveFunctions

    using Lc2ppiKModelLHCb
end

# ╔═╡ 97e2902d-8ea9-4fec-b4d4-25985db069a2
theme(:wong2, frame=:box, grid=false, minorticks=true,
    guidefontvalign=:top, guidefonthalign=:right,
    xlim=(:auto, :auto), ylim=(:auto, :auto),
    lw=1, lab="", colorbar=false)

# ╔═╡ 07a14d52-6e8b-4e31-991d-8cacd576e4f4
begin
    # 1) get isobars
    isobarsinput = readjson(joinpath("..", "data", "isobars.json"))["isobars"]
    #
    isobars = Dict()
    for (key, dict) in isobarsinput
        isobars[key] = buildchain(key, dict)
    end

    # 2) update model parameters
    modelparameters =
        readjson(joinpath("..", "data", "modelparameters.json"))["modelstudies"]

    defaultparameters = first(modelparameters)["parameters"]
    defaultparameters["ArK(892)1"] = "1.0 ± 0.0"
    defaultparameters["AiK(892)1"] = "0.0 ± 0.0"
    #
    shapeparameters = filter(x -> x[1] != 'A', keys(defaultparameters))

    parameterupdates = [ # 6 values are updated
        "K(1430)" => (γ=eval(Meta.parse(defaultparameters["gammaK(1430)"])).val,),
        "K(700)" => (γ=eval(Meta.parse(defaultparameters["gammaK(700)"])).val,),
        "L(1520)" => (m=eval(Meta.parse(defaultparameters["ML(1520)"])).val,
            Γ=eval(Meta.parse(defaultparameters["GL(1520)"])).val),
        "L(2000)" => (m=eval(Meta.parse(defaultparameters["ML(2000)"])).val,
            Γ=eval(Meta.parse(defaultparameters["GL(2000)"])).val)]
    #
    @assert length(shapeparameters) == 6

    # apply updates
    for (p, u) in parameterupdates
        BW = isobars[p].Xlineshape
        isobars[p] = merge(isobars[p], (Xlineshape=updatepars(BW, merge(BW.pars, u)),))
    end

    # 3) get couplings
    couplingkeys = collect(filter(x -> x[1:2] == "Ar", keys(defaultparameters)))
    isobarnames = map(x -> x[3:end-1], couplingkeys)

    const terms = [
        let
            #
            c_re_key = "Ar" * parname[3:end] # = parname
            c_im_key = "Ai" * parname[3:end]
            value_re = eval(Meta.parse(defaultparameters[c_re_key])).val
            value_im = eval(Meta.parse(defaultparameters[c_im_key])).val
            value = value_re + 1im * value_im
            #
            c0, d = parname2decaychain(parname, isobars)
            #
            (c0 * value, d)
        end for parname in couplingkeys
    ]


    const chains = getindex.(terms, 2)
    const couplings = getindex.(terms, 1)
end

# ╔═╡ 05ac73c0-38e0-477a-b373-63993f618d8c
md"""
### Numerical integration over the dataset
"""

# ╔═╡ 684af91d-5314-45d2-ae33-065091e47025
Ai(σs) = [[amplitude(σs, two_λs, d) for d in chains]
          for two_λs in itr(tbs.two_js)]

# ╔═╡ bddbdd76-169a-41f2-ae85-2fd31c4e99f8
const pdata = flatDalitzPlotSample(ms; Nev=100_000);

# ╔═╡ e936b9b0-809c-489a-8231-378220e982c3
Aiv = ThreadsX.map(Ai, pdata);

# ╔═╡ e18dd0c6-7d48-489f-a9ac-b9a1fbc52650
Iv = intensity.(Aiv, Ref(couplings));

# ╔═╡ bf5d7a76-0398-4268-b1ce-6ac545f6816c
I0 = sum(Iv)

# ╔═╡ bd7c24f5-5607-4210-b84a-9ebb8d9ed41a
md"""
### Compute the rate matrix
"""

# ╔═╡ 8294d193-4890-4d09-b174-5f8c75888720
begin
    delta_i(n, iv...) = (v = zeros(n); v[[iv...]] .= 1.0; v)
    nchains = length(couplings)
    ratematrix = zeros(nchains, nchains)
    for i in 1:nchains, j in i:nchains
        cij = delta_i(nchains, i, j) .* couplings
        Iξv = intensity.(Aiv, Ref(cij))
        ratematrix[i, j] = sum(Iξv) / I0 * 100
        ratematrix[j, i] = ratematrix[i, j]
    end
    for i in 1:nchains, j in i+1:nchains
        ratematrix[i, j] += -ratematrix[i, i] - ratematrix[j, j]
        ratematrix[j, i] += -ratematrix[i, i] - ratematrix[j, j]
        ratematrix[i, j] /= 2
        ratematrix[j, i] /= 2
    end
end

# ╔═╡ 4d8d466e-7e90-40b5-b423-9b25a75761cb
@assert sum(ratematrix) ≈ 100

# ╔═╡ 757c2cbb-5967-4af6-b811-79e7901776a8
grouppedchains = getindex.(
    sort(sort(collect(enumerate(isobarnames)),
            by=x -> x[2]),
        by=x -> findfirst(x[2][1], "LDK")), 1);

# ╔═╡ 38d915db-9fa8-400b-953e-fc2750f396c0
let
    grouppedratematrix = ratematrix[grouppedchains, grouppedchains]
    #
    s(two_λ) = iseven(two_λ) ?
               string(div(two_λ, 2)) :
               ("-", "")[1+div((sign(two_λ) + 1), 2)] * "½"
    labelchain(chain) = chain.Xlineshape.name * " " *
                        s(chain.HRk.two_λa) * "," * s(chain.HRk.two_λb)
    labels = labelchain.(chains)[grouppedchains]
    #
    clim = maximum(ratematrix) .* (-1, 1)
    heatmap(grouppedratematrix;
        xticks=(1:nchains, labels), xrotation=90,
        yticks=(1:nchains, labels), aspectratio=1,
        size=(600, 600), c=:delta, colorbar=true,
        title="Rate matrix for chains", clim)
    for i in 1:nchains, j in i+1:nchains
        annotate!((i, j, text(
            grouppedratematrix[i, j] ≥ 0 ? "+" : "-", 4)))
    end
    plot!()
end

# ╔═╡ b275f3ac-65a0-46a8-b375-57fa56d489ef
group(ratematrix, sectors) =
    [sum(getindex(ratematrix, iv, jv))
     for iv in sectors,
     jv in sectors]

# ╔═╡ 79d91b3f-191e-478a-b940-5d896da658a9
let
    isobarnameset = collect(Set(isobarnames))
    sort!(isobarnameset, by=x -> findfirst(x[1], "LDK"))
    #
    sectors = [couplingsmap = collect(1:nchains)[(isobarnames.==s)]
               for s in isobarnameset]
    #
    grouppedratematrix = group(ratematrix, sectors)
    nisobars = length(isobarnameset)
    for i in 1:nisobars, j in i+1:nisobars
        grouppedratematrix[j, i] *= 2
        grouppedratematrix[i, j] = 0
    end
    @assert sum(grouppedratematrix) ≈ 100

    clim = maximum(grouppedratematrix) .* (-1.2, 1.2)
    heatmap(grouppedratematrix;
        xticks=(1:nchains, isobarnameset), xrotation=90,
        yticks=(1:nchains, isobarnameset), aspectratio=1,
        size=(600, 600), c=:delta, colorbar=true,
        title="Rate matrix for isobars", clim)
    for i in 1:nisobars, j in i:nisobars
        annotate!((i, j, text(
            round(grouppedratematrix[j, i], digits=2), 6,
            i == j ? :red : :black)))
    end
    plot!()
end

# ╔═╡ Cell order:
# ╠═03733bd2-dcf3-11ec-231f-8dab0ad6b19e
# ╠═97e2902d-8ea9-4fec-b4d4-25985db069a2
# ╠═07a14d52-6e8b-4e31-991d-8cacd576e4f4
# ╠═05ac73c0-38e0-477a-b373-63993f618d8c
# ╠═684af91d-5314-45d2-ae33-065091e47025
# ╠═bddbdd76-169a-41f2-ae85-2fd31c4e99f8
# ╠═e936b9b0-809c-489a-8231-378220e982c3
# ╠═e18dd0c6-7d48-489f-a9ac-b9a1fbc52650
# ╠═bf5d7a76-0398-4268-b1ce-6ac545f6816c
# ╟─bd7c24f5-5607-4210-b84a-9ebb8d9ed41a
# ╠═8294d193-4890-4d09-b174-5f8c75888720
# ╠═4d8d466e-7e90-40b5-b423-9b25a75761cb
# ╠═757c2cbb-5967-4af6-b811-79e7901776a8
# ╠═38d915db-9fa8-400b-953e-fc2750f396c0
# ╠═b275f3ac-65a0-46a8-b375-57fa56d489ef
# ╠═79d91b3f-191e-478a-b940-5d896da658a9

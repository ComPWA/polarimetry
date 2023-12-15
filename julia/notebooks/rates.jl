### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 03733bd2-dcf3-11ec-231f-8dab0ad6b19e
# ╠═╡ show_logs = false
begin
    cd(joinpath(@__DIR__, ".."))
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
    #
    using YAML
    using Plots
    using LaTeXStrings
    import Plots.PlotMeasures.mm
    using RecipesBase
    #
    using StaticArrays
    using LinearAlgebra
    using Parameters
    using Measurements
    using DataFrames
    using ThreadsX
    #
    using ThreeBodyDecay
    using ThreeBodyDecay.PartialWaveFunctions

    using Lc2ppiKSemileptonicModelLHCb
end

# ╔═╡ 97e2902d-8ea9-4fec-b4d4-25985db069a2
theme(:wong2, frame=:box, grid=false, minorticks=true,
    guidefontvalign=:top, guidefonthalign=:right,
    xlim=(:auto, :auto), ylim=(:auto, :auto),
    lw=1, lab="", colorbar=false)

# ╔═╡ bea43e41-90dd-41cd-8ede-f483c0a2a80e
const model = published_model("Default amplitude model")

# ╔═╡ 05ac73c0-38e0-477a-b373-63993f618d8c
md"""
### Numerical integration over the dataset
"""

# ╔═╡ bddbdd76-169a-41f2-ae85-2fd31c4e99f8
const pdata = flatDalitzPlotSample(ms; Nev=100_000);

# ╔═╡ f7e600be-536e-4c64-9c63-4cb7c3c013ad
const Aiv = ThreadsX.collect(
    SVector([amplitude(d, σs, two_λs) for d in model.chains])
    for two_λs in itr(tbs.two_js), σs in pdata);

# ╔═╡ bf5d7a76-0398-4268-b1ce-6ac545f6816c
I0 = ThreadsX.sum(Aiv) do x
    abs2(sum(x .* model.couplings))
end

# ╔═╡ bd7c24f5-5607-4210-b84a-9ebb8d9ed41a
md"""
### Compute the rate matrix
"""

# ╔═╡ 8294d193-4890-4d09-b174-5f8c75888720
begin
    delta_i(n, iv...) = (v = zeros(n); v[[iv...]] .= 1.0; v)
    nchains = length(model.couplings)
    ratematrix = zeros(nchains, nchains)
    for i in 1:nchains, j in i:nchains
        cij = delta_i(nchains, i, j) .* model.couplings
        Iξv = ThreadsX.sum(Aiv) do x
            abs2(sum(x .* cij))
        end
        ratematrix[i, j] = Iξv / I0 * 100
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

# ╔═╡ 38d915db-9fa8-400b-953e-fc2750f396c0
let
    s(two_λ) = iseven(two_λ) ?
               string(div(two_λ, 2)) :
               ("-", "")[1+div((sign(two_λ) + 1), 2)] * "½"
    labelchain(chain) = chain.Xlineshape.name * " " *
                        s(chain.HRk.two_λa) * "," * s(chain.HRk.two_λb)
    labels = labelchain.(model.chains)
    #
    clim = maximum(ratematrix) .* (-1, 1)
    heatmap(ratematrix;
        xticks=(1:nchains, labels), xrotation=90,
        yticks=(1:nchains, labels), aspectratio=1,
        size=(600, 600), c=:delta, colorbar=true,
        title="Rate matrix for chains", clim)
    for i in 1:nchains, j in i+1:nchains
        annotate!((i, j, text(
            ratematrix[i, j] ≥ 0 ? "+" : "-", 4)))
    end
    plot!()
end

# ╔═╡ 46380761-d719-4ee6-beed-fc96601ed26a
begin
    isobarnameset = collect(Set(model.isobarnames))
    sort!(isobarnameset, by=x -> eval(Meta.parse(x[3:end-1])))
    sort!(isobarnameset, by=x -> findfirst(x[1], "LDK"))

end

# ╔═╡ b275f3ac-65a0-46a8-b375-57fa56d489ef
group(ratematrix, sectors) =
    [sum(getindex(ratematrix, iv, jv))
     for iv in sectors,
     jv in sectors]

# ╔═╡ aafae618-b576-46c5-852e-9dd81b19c10b
grouppedratematrix = let
    sectors = [couplingsmap = collect(1:nchains)[(model.isobarnames.==s)]
               for s in isobarnameset]
    _grouppedratematrix = group(ratematrix, sectors)
    nisobars = length(isobarnameset)
    for i in 1:nisobars, j in i+1:nisobars
        _grouppedratematrix[j, i] *= 2
        _grouppedratematrix[i, j] = 0
    end
    @assert sum(_grouppedratematrix) ≈ 100
    _grouppedratematrix
end;

# ╔═╡ 79d91b3f-191e-478a-b940-5d896da658a9
let
    clim = maximum(grouppedratematrix) .* (-1.2, 1.2)
    heatmap(grouppedratematrix;
        xticks=(1:nchains, isobarnameset), xrotation=90,
        yticks=(1:nchains, isobarnameset), aspectratio=1,
        size=(600, 600), c=:delta, colorbar=true,
        title="Rate matrix for isobars", clim)
    Nξ = length(isobarnameset)
    for i in 1:Nξ, j in i:Nξ
        annotate!((i, j, text(
            round(grouppedratematrix[j, i], digits=2), 6,
            i == j ? :red : :black)))
    end
    plot!()
end

# ╔═╡ 1cf6f963-903d-42ae-af3b-cab2812e7eb9
md"""
### Write output
"""

# ╔═╡ 24a5f0fe-430f-402c-ae08-db0d32f2bc59
begin
    ratesdict = []
    for (k, r) in zip(isobarnameset, diag(grouppedratematrix))
        push!(ratesdict, k => round(r; digits=2))
    end
    results_dir = "results"
    !(results_dir ∈ readdir(".")) && mkdir(results_dir)
    writejson(joinpath(results_dir, "rates.json"),
        Dict("rate" => ratesdict,
            "isobars" => isobarnameset,
            "ratematrix" => round.(grouppedratematrix; digits=2)))
end

# ╔═╡ Cell order:
# ╠═03733bd2-dcf3-11ec-231f-8dab0ad6b19e
# ╠═97e2902d-8ea9-4fec-b4d4-25985db069a2
# ╠═bea43e41-90dd-41cd-8ede-f483c0a2a80e
# ╟─05ac73c0-38e0-477a-b373-63993f618d8c
# ╠═bddbdd76-169a-41f2-ae85-2fd31c4e99f8
# ╠═f7e600be-536e-4c64-9c63-4cb7c3c013ad
# ╠═bf5d7a76-0398-4268-b1ce-6ac545f6816c
# ╟─bd7c24f5-5607-4210-b84a-9ebb8d9ed41a
# ╠═8294d193-4890-4d09-b174-5f8c75888720
# ╠═4d8d466e-7e90-40b5-b423-9b25a75761cb
# ╠═38d915db-9fa8-400b-953e-fc2750f396c0
# ╠═46380761-d719-4ee6-beed-fc96601ed26a
# ╠═b275f3ac-65a0-46a8-b375-57fa56d489ef
# ╠═aafae618-b576-46c5-852e-9dd81b19c10b
# ╠═79d91b3f-191e-478a-b940-5d896da658a9
# ╟─1cf6f963-903d-42ae-af3b-cab2812e7eb9
# ╠═24a5f0fe-430f-402c-ae08-db0d32f2bc59

### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ dea88954-c7b2-11ec-1a3d-717739cfd08b
# ╠═╡ show_logs = false
begin
    cd(joinpath(@__DIR__, ".."))
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
    #
    import YAML
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
    using StaticArrays
    #
    using ThreeBodyDecay
    using ThreeBodyDecay.PartialWaveFunctions

    using Lc2ppiKSemileptonicModelLHCb
end

# ╔═╡ d3cb114e-ee0c-4cd5-87fb-82289849aceb
md"""
# LHCb model for $\Lambda_c^+ \to p K^- \pi^+$ decays
"""

# ╔═╡ e04157cc-7697-41e5-8e4e-3556332929ef
theme(:wong2, frame=:box, grid=false, minorticks=true,
    guidefontvalign=:top, guidefonthalign=:right,
    xlim=(:auto, :auto), ylim=(:auto, :auto),
    lw=1, lab="", colorbar=false,
    xlab=L"m^2(K\pi)\,\,(\mathrm{GeV}^2)",
    ylab=L"m^2(pK)\,\,(\mathrm{GeV}^2)")

# ╔═╡ 877f6aab-d0ff-44d7-9ce0-e43d895be297
md"""
## Parsing the isobar description
"""

# ╔═╡ 7cc4c5f9-4392-4b57-88af-59d1cf308162
(; modelparameters, particledict) = expose_model_description()

# ╔═╡ 78032115-badb-45b6-b20f-15496d460d57
modeldict = modelparameters["Default amplitude model"]

# ╔═╡ 3f9e1ed7-b308-4cbe-805d-cbe3f22e3d1f
defaultparameters = modeldict["parameters"]

# ╔═╡ b0f5c181-dcb2-48f8-a510-57eac44ca4d9
begin
    isobars = Dict()
    for (key, lineshape) in modeldict["lineshapes"]
        dict = Dict{String,Any}(particledict[key])
        dict["lineshape"] = lineshape
        isobars[key] = definechaininputs(key, dict)
    end
end;

# ╔═╡ 98cca824-73db-4e1c-95ae-68c3bc8574fe
md"""
#### Summary of the parsed data
"""

# ╔═╡ c7572ffb-c4c7-4ce6-90a9-8237214ac91b
let
    isoprint = DataFrame()
    for (k, v) in isobars
        @unpack Xlineshape, two_s, parity = v
        @unpack l, minL = Xlineshape
        m, Γ = Xlineshape.pars
        push!(isoprint, (; name=k,
            jp=(isodd(two_s) ? "$(two_s)/2" : "$(div(two_s,2))") * parity,
            m, Γ, l, minL,
            lineshape=typeof(Xlineshape).name.name))
    end
    transform!(isoprint, [:m, :Γ] .=> x -> 1e3 * x; renamecols=false)
    sort(isoprint, [
        order(:name, by=x -> findfirst(x[1], "LDK")),
        order(:m)])
end

# ╔═╡ cee9dc28-8048-49e7-8caf-8e07bcd884c4
let
    plot(layout=grid(1, 3), size=(1000, 240), left_margin=4mm, bottom_margin=7mm)
    for (k, v) in isobars
        plot!(sp=v.k, v.Xlineshape, lab=k,
            xlab=L"m^2\,\,(\mathrm{GeV}^2)", ylab=L"(\mathrm{a.u.})")
    end
    plot!()
end

# ╔═╡ 30c3c8ef-ad69-43e6-9a75-525dfbf7007a
md"""
### Fit parameters
"""

# ╔═╡ f9158b4d-4d27-4aba-bf5a-529135ec48e2
shapeparameters = filter(x -> x[1] != 'A', keys(defaultparameters))

# ╔═╡ 838d0391-f1f2-4a4f-9b8f-0a6021720c5b
x = defaultparameters["gammaK(1430)"]

# ╔═╡ 1db41980-ea36-4238-90cd-bf2427772ea9
parameterupdates = [
    "K(1430)" => (γ=MeasuredParameter(defaultparameters["gammaK(1430)"]).val,),
    "K(700)" => (γ=MeasuredParameter(defaultparameters["gammaK(700)"]).val,),
    "L(1520)" => (m=MeasuredParameter(defaultparameters["ML(1520)"]).val,
        Γ=MeasuredParameter(defaultparameters["GL(1520)"]).val),
    "L(2000)" => (m=MeasuredParameter(defaultparameters["ML(2000)"]).val,
        Γ=MeasuredParameter(defaultparameters["GL(2000)"]).val)]

# ╔═╡ 1f6dc202-8734-4232-9f48-a88ebf17ff93
for (p, u) in parameterupdates
    BW = isobars[p].Xlineshape
    isobars[p] = merge(isobars[p], (Xlineshape=updatepars(BW, merge(BW.pars, u)),))
end

# ╔═╡ 21c125fc-e218-4676-9846-822d951f4f1b
md"""
### Coupligns
"""

# ╔═╡ 64a1d1f5-da82-4ce6-8731-ccc7439bd881
couplingkeys = collect(filter(x -> x[1:2] == "Ar", keys(defaultparameters)))

# ╔═╡ 168707da-d42c-4b2f-94b6-f7bc15cb29cb
const terms = [
    let
        #
        c_re_key = "Ar" * parname[3:end] # = parname
        c_im_key = "Ai" * parname[3:end]
        value_re = MeasuredParameter(defaultparameters[c_re_key]).val
        value_im = MeasuredParameter(defaultparameters[c_im_key]).val
        value = value_re + 1im * value_im
        #
        c0, d = parname2decaychain(parname, isobars)
        #
        (c0 * value, d)
    end for parname in couplingkeys
]

# ╔═╡ 8d8824e8-7809-4368-840c-b8f6d38ad7c2
const model =
    Lc2ppiKModel(;
        chains=getindex.(terms, 2),
        couplings=getindex.(terms, 1),
        isobarnames=map(x -> x[3:end-1], couplingkeys))

# ╔═╡ d547fc89-3756-49d3-8b49-ad0c56c5c3a3
md"""
### Plotting the distributions
"""

# ╔═╡ b833843f-c8b6-44c6-847c-efce0ed989d4
plot(ms, Base.Fix1(unpolarizedintensity, model), iσx=1, iσy=2, Ngrid=54)

# ╔═╡ ed7f802c-a931-4114-aa30-6d10735520d7
md"""
### Polarization sensitivity
"""

# ╔═╡ 45843dac-096a-4bd5-82eb-aad4f18b8f86
begin
    import Lc2ppiKSemileptonicModelLHCb: expectation
    expectation(Op, σs) = sum(
        conj(amplitude(model, σs, [two_λ, 0, 0, two_ν′])) *
        Op[twoλ2ind(two_ν′), twoλ2ind(two_ν)] *
        amplitude(model, σs, [two_λ, 0, 0, two_ν])
        for two_λ in [-1, 1], two_ν in [-1, 1], two_ν′ in [-1, 1]) |> real
end

# ╔═╡ 2f42ac46-554c-4376-83a9-ad8eeaf90422
function Ialphaongrid(; iσx, iσy, Ngrid=100)
    σxv = range(lims(iσx, ms)..., length=Ngrid)
    σyv = range(lims(iσy, ms)..., length=Ngrid)
    #
    σsv = [
        ThreeBodyDecay.invs(σx, σy; iσx=iσx, iσy=iσy, ms=ms)
        for σy in σyv, σx in σxv]
    Iv = [
        map(σs -> (Kibble(σs, ms^2) < 0 ? expectation(Op, σs) : NaN), σsv)
        for Op in σPauli]
    return (; σxv, σyv, iσx, iσy, Iv)
end

# ╔═╡ 190a6f02-bd33-447c-a5b3-4dd3dd79579c
dataIalphaongrid = Ialphaongrid(; iσx=1, iσy=2, Ngrid=55);

# ╔═╡ b50627f0-d3a6-4931-867e-5d102b543502
@recipe function f(nt::NamedTuple{(:σxv, :σyv, :iσx, :iσy, :Iv)})
    @unpack σxv, σyv, Iv = nt
    layout := grid(1, 3)
    size --> (900, 300)
    for (i, t) in enumerate([L"\alpha_x", L"\alpha_y", L"\alpha_z"])
        @series begin
            subplot := i
            title := t
            clim --> (-1, 1)
            c --> :balance
            seriestype := :heatmap
            calv = Iv[i] ./ Iv[4]
            (σxv, σyv, calv)
        end
    end
end

# ╔═╡ c1554cf8-3cfa-4209-9ef1-31f424ebb361
md"""
The three-body decay is sensitive to the inital polarization even if the Dalitz plot distribution is integated over.
In this case the differential decay rate is a function of the three production angles, $\phi$, $\theta$, and $\chi$. The asymmetries are given by determined by the effective values of $\alpha$.

$I(\phi_1, \theta_1, \chi_1) = I_0\left(1+\sum_{i,j}P_i R_{ij}(\phi, \theta, \chi) \overline{\alpha}_j\right)\,$

The averaging over the dalitz plot variables does the averaging over momenta and the angles between $p$, $K$, and $\pi$ in the decay plane.
Importnatly, the choice of the alignment configuration is determining which of the final-state particles is an anchor. Expectedly, the averaged value of the asymmetry vector is different for the three possible alignments.
"""

# ╔═╡ 26fa6889-03e3-42b5-8222-c15f0ab3caa2
md"""
#### Alignment (1)
"""

# ╔═╡ ef85ad9b-13bb-45bc-985e-289a4a81fe7f
plot(dataIalphaongrid, left_margin=3mm, bottom_margin=5mm)

# ╔═╡ a2cbe50c-c1a1-4572-b440-e3c618146ae4
ᾱ⁽¹⁾ = sum.(filter.(!isnan, dataIalphaongrid.Iv[1:3])) ./
        sum(filter(!isnan, dataIalphaongrid.Iv[4]));

# ╔═╡ 25466667-73de-4dcb-a97e-592dac225c10
md"""
#### ᾱ⁽¹⁾ = $(string(round.(ᾱ⁽¹⁾, digits=3)))
"""

# ╔═╡ f41c087e-580d-408e-afa1-50d013ffa47f
md"""
Now, rotate $R_y^{-1}(\zeta^0_{x(1)})$ for chains 2 (x=2) and 3 (x=3)
"""

# ╔═╡ 14df1109-1904-4536-9f15-3009e4003a7f
function rotationongrid(; iσx, iσy, Ngrid=100)
    σxv = range(lims(iσx, ms)..., length=Ngrid)
    σyv = range(lims(iσy, ms)..., length=Ngrid)
    σsv = [
        ThreeBodyDecay.invs(σx, σy; iσx=iσx, iσy=iσy, ms=ms)
        for σy in σyv, σx in σxv]
    cosζ⁰₁₂, cosζ⁰₃₁ = [
        map(σs -> (Kibble(σs, ms^2) < 0 ? cosζ(r, σs, ms^2) : NaN), σsv)
        for r in [wr(1, 2, 0), wr(3, 1, 0)]]
    return (; σxv, σyv, iσx, iσy, cosζ⁰₁₂, cosζ⁰₃₁)
end

# ╔═╡ 4cec3154-2b51-4355-a813-2ce95e05eeae
datarotationongrid = let
    @unpack iσx, iσy, σxv = dataIalphaongrid
    Ngrid = length(σxv)
    rotationongrid(; iσx, iσy, Ngrid)
end;

# ╔═╡ c327efc1-8e7f-49e7-9407-207a44647d6f
O3rotate(α1, α2, cosθ, signθ) =
    α1 * cosθ - α2 * signθ * sqrt(1 - cosθ^2)

# ╔═╡ bec3ddaa-c3d2-47ae-9208-359f08a85353
md"""
#### Alignment (2)
"""

# ╔═╡ 3c7b043b-0d58-4e1b-8ce7-52bf65c6ff6f
function O3rotate!(Iv, coswv, signθ)
    for ci in CartesianIndices(coswv)
        Iv[3][ci], Iv[1][ci] =
            O3rotate(Iv[3][ci], Iv[1][ci], coswv[ci], signθ),
            O3rotate(Iv[1][ci], Iv[3][ci], coswv[ci], -signθ)
    end
end

# ╔═╡ d7227cb9-02c3-43da-b44b-5af38afbe0b9
begin
    dataIalphaongrid_rot12 = merge(dataIalphaongrid,
        (; Iv=copy.(dataIalphaongrid.Iv)))
    O3rotate!(dataIalphaongrid_rot12.Iv, datarotationongrid.cosζ⁰₁₂, +1)
end

# ╔═╡ 320828c8-5b5d-43d0-9883-e54354296488
plot(dataIalphaongrid_rot12, left_margin=3mm, bottom_margin=5mm)

# ╔═╡ 9353e902-1c7b-487e-a5f4-3e99bf0fada8
ᾱ⁽²⁾ = sum.(filter.(!isnan, dataIalphaongrid_rot12.Iv[1:3])) ./
        sum(filter(!isnan, dataIalphaongrid_rot12.Iv[4]))

# ╔═╡ 264d4fed-d656-48f2-be04-8ef0122721de
md"""
#### ᾱ⁽²⁾ = $(string(round.(ᾱ⁽²⁾, digits=3)))
"""

# ╔═╡ ddcadf04-1842-4218-9ca0-550d9ca48f73
md"""
#### Alignment (3)
"""

# ╔═╡ a1018b19-83ff-40a1-b5ec-2f038fc0f981
begin
    dataIalphaongrid_rot31 = merge(dataIalphaongrid,
        (; Iv=copy.(dataIalphaongrid.Iv)))
    O3rotate!(dataIalphaongrid_rot31.Iv, datarotationongrid.cosζ⁰₃₁, -1)
end

# ╔═╡ 592595bc-dbe4-4f8c-833a-76790d3a00f8
plot(dataIalphaongrid_rot31, left_margin=4mm, bottom_margin=5.5mm)

# ╔═╡ 3a4be1b4-a544-437a-b1f6-bd82819ba676
ᾱ⁽³⁾ = sum.(filter.(!isnan, dataIalphaongrid_rot31.Iv[1:3])) ./
        sum(filter(!isnan, dataIalphaongrid_rot31.Iv[4]));

# ╔═╡ ac7e83fb-1ab0-41a0-ae78-c9a36053d149
md"""
#### ᾱ⁽³⁾ = $(string(round.(ᾱ⁽³⁾, digits=3)))
"""

# ╔═╡ e550eec0-6911-4016-824f-b21f82d10e9a
DataFrame(
    :alignment => "confuguration " .* string.([1, 2, 3]),
    :components => NamedTuple{(:ᾱx, :ᾱy, :ᾱz)}.(
        [ᾱ⁽¹⁾, ᾱ⁽²⁾, ᾱ⁽³⁾] .|> x -> round.(x; digits=3))
) |> df -> select(df, Not(:components),
    :components => AsTable,
    :components => ByRow(norm) => :norm)

# ╔═╡ 8339a757-9b66-4cb9-9bbb-a7de466fc3bf
md"""
Interestingly, the value of the averaged asymmetry parameter is the highest for the alignment with respect to the first chain, i.e. $K^*$ resonances.
The reason is likely to be that the $z$-axis is determined by the vector of $K^*$ which is opposite to the vector of the proton in $\Lambda_c^+$ rest frame.
"""

# ╔═╡ b27001c0-df6c-4a47-ae53-8cee96cbf984
md"""
### Numerical projection
"""

# ╔═╡ 88c58ce2-c98b-4b60-901f-ed95099c144b
pdata = flatDalitzPlotSample(ms; Nev=300_000);

# ╔═╡ 589c3c6f-fc24-4c2c-bf42-df0b3187d8cf
const Aiv = ThreadsX.collect(
    SVector([amplitude(d, σs, two_λs) for d in model.chains])
    for two_λs in itr(tbs.two_js), σs in pdata);

# ╔═╡ 25a293e0-05c6-44d2-bec0-42649558e1c2
Iαv = [sum(
    sum(Aiv[iλ, 1, 1, iν′, e] .* model.couplings)' *
    σP[3-iν′, 3-iν] *
    sum(Aiv[iλ, 1, 1, iν, e] .* model.couplings)
    for iλ in [1, 2], iν in [1, 2], iν′ in [1, 2],
    e in 1:size(Aiv, 5)) |> real for σP in σPauli]

# ╔═╡ 01bbaad2-81e9-436c-b698-d1a16f9da529
md"""
#### Cross-check: ᾱ⁽¹⁾ = $(string(round.(Iαv[1:3] ./ Iαv[4], digits=3)))
"""

# ╔═╡ b3c0d027-4c98-47e4-9f9a-77ba1d10ae98
Iv = sum(Aiv; dims=(1, 2, 3, 4)) do x
    abs2(sum(x .* model.couplings))
end[1, 1, 1, 1, :]

# ╔═╡ 50d4a601-3f14-4034-9e6f-08eae9ca7d7c
I0 = sum(Iv)

# ╔═╡ 8e06d5ec-4b98-441d-919b-7b90071e6674
histogram2d(size=(500, 440),
    getproperty.(pdata, :σ1),
    getproperty.(pdata, :σ2), weights=Iv, bins=100)

# ╔═╡ fc62283e-8bcb-4fd1-8809-b7abeb991030
begin
    bins = 150
    plot(layout=grid(1, 3), size=(800, 300), yaxis=nothing,
        ylab=L"\mathrm{rate}\,\,(\mathrm{a.u.})",
        stephist(getproperty.(pdata, :σ2), weights=Iv; bins,
            xlab=L"m^2(pK^-)\,(\mathrm{GeV}^2)"),
        stephist(getproperty.(pdata, :σ1), weights=Iv; bins,
            xlab=L"m^2(K^-\pi^+)\,(\mathrm{GeV}^2)"),
        stephist(getproperty.(pdata, :σ3), weights=Iv; bins,
            xlab=L"m^2(p\pi^-)\,(\mathrm{GeV}^2)"),
        left_margin=5mm, bottom_margin=7mm)
    for s in Set(model.isobarnames)
        couplings = model.couplings .* (model.isobarnames .== s)
        Iξv = sum(Aiv; dims=(1, 2, 3, 4)) do x
            abs2(sum(x .* couplings))
        end[1, 1, 1, 1, :]
        #
        stephist!(sp=1, getproperty.(pdata, :σ2), weights=Iξv; bins, lab="")
        stephist!(sp=2, getproperty.(pdata, :σ1), weights=Iξv; bins, lab="")
        stephist!(sp=3, getproperty.(pdata, :σ3), weights=Iξv; bins, lab=s)
    end
    plot!()
end

# ╔═╡ 0a976167-b074-4694-ab97-aecfcd67cc25
begin
    sξnames = collect(Set(model.isobarnames)) |>
              c -> sort(c; by=x -> eval(Meta.parse(x[3:end-1]))) |>
                   c -> sort(c; by=x -> findfirst(x[1], "LDK"))
    #
    rates = DataFrame()
    for s in sξnames
        couplingsmap = (model.isobarnames .== s)
        couplings = model.couplings .* couplingsmap
        Iξv = sum(Aiv; dims=(1, 2, 3, 4)) do x
            abs2(sum(x .* couplings))
        end[1, 1, 1, 1, :]
        #
        cs = couplings[couplingsmap]
        Hs = getproperty.(model.chains[couplingsmap], :HRk)
        #
        ex, ey, ez, e0 = expectation.(σPauli, Ref(cs), Ref(Hs))
        #
        αx, αy, αz = (ex, ey, ez) ./ e0
        push!(rates, (isobarname=s, rate=sum(Iξv) / I0 * 100,
            αz, αy, αx, α_abs=sqrt(αz^2 + αy^2 + αx^2)))
    end
    #
    transform(rates, [:rate, :αz, :α_abs] .=> ByRow(x -> round(x; digits=2));
        renamecols=false)
end

# ╔═╡ Cell order:
# ╟─d3cb114e-ee0c-4cd5-87fb-82289849aceb
# ╠═dea88954-c7b2-11ec-1a3d-717739cfd08b
# ╠═e04157cc-7697-41e5-8e4e-3556332929ef
# ╟─877f6aab-d0ff-44d7-9ce0-e43d895be297
# ╠═7cc4c5f9-4392-4b57-88af-59d1cf308162
# ╠═78032115-badb-45b6-b20f-15496d460d57
# ╠═3f9e1ed7-b308-4cbe-805d-cbe3f22e3d1f
# ╠═b0f5c181-dcb2-48f8-a510-57eac44ca4d9
# ╟─98cca824-73db-4e1c-95ae-68c3bc8574fe
# ╠═c7572ffb-c4c7-4ce6-90a9-8237214ac91b
# ╠═cee9dc28-8048-49e7-8caf-8e07bcd884c4
# ╟─30c3c8ef-ad69-43e6-9a75-525dfbf7007a
# ╠═f9158b4d-4d27-4aba-bf5a-529135ec48e2
# ╠═838d0391-f1f2-4a4f-9b8f-0a6021720c5b
# ╠═1db41980-ea36-4238-90cd-bf2427772ea9
# ╠═1f6dc202-8734-4232-9f48-a88ebf17ff93
# ╟─21c125fc-e218-4676-9846-822d951f4f1b
# ╠═64a1d1f5-da82-4ce6-8731-ccc7439bd881
# ╠═168707da-d42c-4b2f-94b6-f7bc15cb29cb
# ╠═8d8824e8-7809-4368-840c-b8f6d38ad7c2
# ╟─d547fc89-3756-49d3-8b49-ad0c56c5c3a3
# ╠═b833843f-c8b6-44c6-847c-efce0ed989d4
# ╟─ed7f802c-a931-4114-aa30-6d10735520d7
# ╠═45843dac-096a-4bd5-82eb-aad4f18b8f86
# ╠═2f42ac46-554c-4376-83a9-ad8eeaf90422
# ╠═190a6f02-bd33-447c-a5b3-4dd3dd79579c
# ╠═b50627f0-d3a6-4931-867e-5d102b543502
# ╟─c1554cf8-3cfa-4209-9ef1-31f424ebb361
# ╟─26fa6889-03e3-42b5-8222-c15f0ab3caa2
# ╠═ef85ad9b-13bb-45bc-985e-289a4a81fe7f
# ╠═a2cbe50c-c1a1-4572-b440-e3c618146ae4
# ╟─25466667-73de-4dcb-a97e-592dac225c10
# ╟─f41c087e-580d-408e-afa1-50d013ffa47f
# ╠═14df1109-1904-4536-9f15-3009e4003a7f
# ╠═4cec3154-2b51-4355-a813-2ce95e05eeae
# ╠═c327efc1-8e7f-49e7-9407-207a44647d6f
# ╟─bec3ddaa-c3d2-47ae-9208-359f08a85353
# ╠═3c7b043b-0d58-4e1b-8ce7-52bf65c6ff6f
# ╠═d7227cb9-02c3-43da-b44b-5af38afbe0b9
# ╠═320828c8-5b5d-43d0-9883-e54354296488
# ╠═9353e902-1c7b-487e-a5f4-3e99bf0fada8
# ╟─264d4fed-d656-48f2-be04-8ef0122721de
# ╟─ddcadf04-1842-4218-9ca0-550d9ca48f73
# ╠═a1018b19-83ff-40a1-b5ec-2f038fc0f981
# ╠═592595bc-dbe4-4f8c-833a-76790d3a00f8
# ╠═3a4be1b4-a544-437a-b1f6-bd82819ba676
# ╟─ac7e83fb-1ab0-41a0-ae78-c9a36053d149
# ╠═e550eec0-6911-4016-824f-b21f82d10e9a
# ╟─8339a757-9b66-4cb9-9bbb-a7de466fc3bf
# ╟─b27001c0-df6c-4a47-ae53-8cee96cbf984
# ╠═88c58ce2-c98b-4b60-901f-ed95099c144b
# ╠═589c3c6f-fc24-4c2c-bf42-df0b3187d8cf
# ╠═25a293e0-05c6-44d2-bec0-42649558e1c2
# ╟─01bbaad2-81e9-436c-b698-d1a16f9da529
# ╠═b3c0d027-4c98-47e4-9f9a-77ba1d10ae98
# ╠═50d4a601-3f14-4034-9e6f-08eae9ca7d7c
# ╠═8e06d5ec-4b98-441d-919b-7b90071e6674
# ╠═fc62283e-8bcb-4fd1-8809-b7abeb991030
# ╠═0a976167-b074-4694-ab97-aecfcd67cc25

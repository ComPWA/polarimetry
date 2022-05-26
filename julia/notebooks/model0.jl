### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ dea88954-c7b2-11ec-1a3d-717739cfd08b
begin
	cd(joinpath(@__DIR__,".."))
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

	using Lb2ppiKModelLHCb
end

# ╔═╡ d3cb114e-ee0c-4cd5-87fb-82289849aceb
md"""
# LHCb model for $\Lambda_c^+ \to p K^- \pi^+$ decays
"""

# ╔═╡ e04157cc-7697-41e5-8e4e-3556332929ef
theme(:wong2, frame=:box, grid=false, minorticks=true,
    guidefontvalign=:top, guidefonthalign=:right,
    xlim=(:auto,:auto), ylim=(:auto,:auto),
    lw=1, lab="", colorbar=false)

# ╔═╡ 877f6aab-d0ff-44d7-9ce0-e43d895be297
md"""
## Parsing the isobar description
"""

# ╔═╡ 7cc4c5f9-4392-4b57-88af-59d1cf308162
isobarsinput = readjson(joinpath("..","data","isobars.json"))["isobars"];

# ╔═╡ b0f5c181-dcb2-48f8-a510-57eac44ca4d9
begin
	isobars = Dict()
	for (key, dict) in isobarsinput
		isobars[key] = buildchain(key, dict)
	end
end ;

# ╔═╡ 98cca824-73db-4e1c-95ae-68c3bc8574fe
md"""
#### Summary of the parsed data
"""

# ╔═╡ c7572ffb-c4c7-4ce6-90a9-8237214ac91b
let
	isoprint = DataFrame()
	for (k,v) in isobars
		@unpack Xlineshape, two_s, parity = v
		@unpack l, minL = Xlineshape
		m,Γ = Xlineshape.pars
		push!(isoprint, (; name=k,
			jp=(isodd(two_s) ? "$(two_s)/2" : "$(div(two_s,2))")*parity,
			m, Γ, l, minL,
			lineshape=typeof(Xlineshape).name.name))
	end
	transform!(isoprint, [:m, :Γ] .=> x->1e3*x; renamecols=false)
	sort(isoprint, [
		order(:name, by=x->findfirst(x[1],"LDK")),
		order(:m)])
end

# ╔═╡ cee9dc28-8048-49e7-8caf-8e07bcd884c4
let
	plot(layout=grid(1,3), size=(1000,240))
	for (k,v) in isobars
		plot!(sp=v.k, v.Xlineshape, lab=k)
	end
	plot!()
end

# ╔═╡ 30c3c8ef-ad69-43e6-9a75-525dfbf7007a
md"""
### Fit parameters
"""

# ╔═╡ 7ecc65a4-9fe7-4209-ad54-f1c8abe52ee5
 modelparameters =
	 readjson(joinpath("..","data","modelparameters.json"))["modelstudies"] ;

# ╔═╡ 24d051e5-4d9e-48c8-aace-225f3a0218cb
begin
	defaultparameters = first(modelparameters)["parameters"]
	defaultparameters["ArK(892)1"] = "1.0 ± 0.0"
	defaultparameters["AiK(892)1"] = "0.0 ± 0.0"
end

# ╔═╡ f9158b4d-4d27-4aba-bf5a-529135ec48e2
shapeparameters = filter(x->x[1]!='A', keys(defaultparameters))

# ╔═╡ 1db41980-ea36-4238-90cd-bf2427772ea9
parameterupdates = [
"K(1430)" => (γ=eval(Meta.parse(defaultparameters["gammaK(1430)"])).val,),
"K(700)" => (γ=eval(Meta.parse(defaultparameters["gammaK(700)"])).val,),
"L(1520)" => (m=eval(Meta.parse(defaultparameters["ML(1520)"])).val,
			  Γ=eval(Meta.parse(defaultparameters["GL(1520)"])).val),
"L(2000)" => (m=eval(Meta.parse(defaultparameters["ML(2000)"])).val,
			  Γ=eval(Meta.parse(defaultparameters["GL(2000)"])).val)]

# ╔═╡ 1f6dc202-8734-4232-9f48-a88ebf17ff93
for (p,u) in parameterupdates
	BW = isobars[p].Xlineshape
	isobars[p] = merge(isobars[p], (Xlineshape=updatepars(BW, merge(BW.pars,u)),))
end

# ╔═╡ 21c125fc-e218-4676-9846-822d951f4f1b
md"""
### Coupligns
"""

# ╔═╡ 64a1d1f5-da82-4ce6-8731-ccc7439bd881
couplingkeys = collect(filter(x->x[1:2]=="Ar", keys(defaultparameters)))

# ╔═╡ 168707da-d42c-4b2f-94b6-f7bc15cb29cb
const terms = [let
	#
	c_re_key = "Ar" * parname[3:end] # = parname
	c_im_key = "Ai" * parname[3:end]
	value_re = eval(Meta.parse(defaultparameters[c_re_key])).val
	value_im = eval(Meta.parse(defaultparameters[c_im_key])).val
	value = value_re + 1im*value_im
	#
	c0, d = parname2decaychain(parname, isobars)
	#
	(c0*value, d)
end for parname in couplingkeys]

# ╔═╡ 8d8824e8-7809-4368-840c-b8f6d38ad7c2
begin
	const chains = getindex.(terms,2)
	const couplings = getindex.(terms,1)
end;

# ╔═╡ d547fc89-3756-49d3-8b49-ad0c56c5c3a3
md"""
### Plotting the distributions
"""

# ╔═╡ abdeb1ac-19dc-45b2-94dd-5e64fb3d8f14
A(σs,two_λs) = sum(c*amplitude(σs,two_λs,d) for (c,d) in zip(couplings,chains))

# ╔═╡ e50afc73-d0a6-4a8d-ae06-c95c9946998d
const I = summed_over_polarization(abs2 ∘ A, tbs.two_js)

# ╔═╡ b833843f-c8b6-44c6-847c-efce0ed989d4
plot(ms,σs->I(σs), iσx=2, iσy=1, Ngrid=54)

# ╔═╡ b27001c0-df6c-4a47-ae53-8cee96cbf984
md"""
### Numerical projection
"""

# ╔═╡ 0736dd22-89dd-4d7f-b332-b0767180ad43
Ai(σs) = [[amplitude(σs,two_λs, d) for d in chains]
	for two_λs in itr(tbs.two_js)]

# ╔═╡ 88c58ce2-c98b-4b60-901f-ed95099c144b
pdata = flatDalitzPlotSample(ms; Nev=100_000) ;

# ╔═╡ 589c3c6f-fc24-4c2c-bf42-df0b3187d8cf
Aiv = ThreadsX.map(Ai, pdata) ;

# ╔═╡ b3c0d027-4c98-47e4-9f9a-77ba1d10ae98
Iv = intensity.(Aiv, Ref(couplings));

# ╔═╡ 50d4a601-3f14-4034-9e6f-08eae9ca7d7c
I0 = sum(Iv)

# ╔═╡ 8e06d5ec-4b98-441d-919b-7b90071e6674
histogram2d(
	getproperty.(pdata, :σ2),
	getproperty.(pdata, :σ1), weights=Iv, bins=100)

# ╔═╡ 78dba088-6d5b-4e4b-a664-f176f9e2d673
isobarnames = map(x->x[3:end-1], couplingkeys)

# ╔═╡ fc62283e-8bcb-4fd1-8809-b7abeb991030
begin
	bins = 150
	plot(layout=grid(1,3), size=(700,350), yaxis=nothing,
		stephist(getproperty.(pdata, :σ2), weights=Iv; bins,
			xlab=L"m^2(pK^-)\,(\mathrm{GeV}^2)"),
		stephist(getproperty.(pdata, :σ1), weights=Iv; bins,
			xlab=L"m^2(K^-\pi^+)\,(\mathrm{GeV}^2)"),
		stephist(getproperty.(pdata, :σ3), weights=Iv; bins,
			xlab=L"m^2(p\pi^-)\,(\mathrm{GeV}^2)"),
	bottom_margin=7mm)
	for s in Set(isobarnames)
		couplingsmap = (isobarnames .== s)
		Iξv = intensity.(Aiv, Ref(couplings .* couplingsmap));
		#
		stephist!(sp=1, getproperty.(pdata, :σ2), weights=Iξv; bins, lab="")
		stephist!(sp=2, getproperty.(pdata, :σ1), weights=Iξv; bins, lab="")
		stephist!(sp=3, getproperty.(pdata, :σ3), weights=Iξv; bins, lab=s)
	end
	plot!()
end

# ╔═╡ 0a976167-b074-4694-ab97-aecfcd67cc25
begin
	rates = DataFrame()
	for s in Set(isobarnames)
		couplingsmap = (isobarnames .== s)
		Iξv = intensity.(Aiv, Ref(couplings .* couplingsmap));
		# 
		cs = couplings[couplingsmap]
		Hs = getproperty.(chains[couplingsmap], :HRk)
		#
		ex, ey, ez, e0 = expectation.(σPauli, Ref(cs), Ref(Hs))
		#
		αx, αy, αz = (ex,ey,ez) ./ e0
		push!(rates, (isobarname=s, rate = sum(Iξv) / I0 * 100,
			αz, αy, αx, α_abs = sqrt(αz^2+αy^2+αx^2)))
	end
	sort(
		sort(
			transform(rates, [:rate, :αz, :α_abs] .=> ByRow(x->round(x; digits=2)); renamecols=false),
			order(:isobarname, by=x->eval(Meta.parse(x[3:end-1])))),
		order(:isobarname, by=x->findfirst(x[1],"LDK")))
end

# ╔═╡ Cell order:
# ╟─d3cb114e-ee0c-4cd5-87fb-82289849aceb
# ╠═dea88954-c7b2-11ec-1a3d-717739cfd08b
# ╠═e04157cc-7697-41e5-8e4e-3556332929ef
# ╟─877f6aab-d0ff-44d7-9ce0-e43d895be297
# ╠═7cc4c5f9-4392-4b57-88af-59d1cf308162
# ╠═b0f5c181-dcb2-48f8-a510-57eac44ca4d9
# ╟─98cca824-73db-4e1c-95ae-68c3bc8574fe
# ╠═c7572ffb-c4c7-4ce6-90a9-8237214ac91b
# ╠═cee9dc28-8048-49e7-8caf-8e07bcd884c4
# ╟─30c3c8ef-ad69-43e6-9a75-525dfbf7007a
# ╠═7ecc65a4-9fe7-4209-ad54-f1c8abe52ee5
# ╠═24d051e5-4d9e-48c8-aace-225f3a0218cb
# ╠═f9158b4d-4d27-4aba-bf5a-529135ec48e2
# ╠═1db41980-ea36-4238-90cd-bf2427772ea9
# ╠═1f6dc202-8734-4232-9f48-a88ebf17ff93
# ╟─21c125fc-e218-4676-9846-822d951f4f1b
# ╠═64a1d1f5-da82-4ce6-8731-ccc7439bd881
# ╠═168707da-d42c-4b2f-94b6-f7bc15cb29cb
# ╠═8d8824e8-7809-4368-840c-b8f6d38ad7c2
# ╟─d547fc89-3756-49d3-8b49-ad0c56c5c3a3
# ╠═abdeb1ac-19dc-45b2-94dd-5e64fb3d8f14
# ╠═e50afc73-d0a6-4a8d-ae06-c95c9946998d
# ╠═b833843f-c8b6-44c6-847c-efce0ed989d4
# ╟─b27001c0-df6c-4a47-ae53-8cee96cbf984
# ╠═0736dd22-89dd-4d7f-b332-b0767180ad43
# ╠═88c58ce2-c98b-4b60-901f-ed95099c144b
# ╠═589c3c6f-fc24-4c2c-bf42-df0b3187d8cf
# ╠═b3c0d027-4c98-47e4-9f9a-77ba1d10ae98
# ╠═50d4a601-3f14-4034-9e6f-08eae9ca7d7c
# ╠═8e06d5ec-4b98-441d-919b-7b90071e6674
# ╠═78dba088-6d5b-4e4b-a664-f176f9e2d673
# ╠═fc62283e-8bcb-4fd1-8809-b7abeb991030
# ╠═0a976167-b074-4694-ab97-aecfcd67cc25

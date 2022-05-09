### A Pluto.jl notebook ###
# v0.18.0

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

# ╔═╡ 24d2ac10-b474-472d-827b-c353f44d9e8c
begin
	const mK = 0.493677
	const mπ = 0.13957018
	const mp = 0.938272046
	const mΣ = 1.18937
	const mΛc = 2.28646
end

# ╔═╡ cfe26457-a531-4fc4-8eee-34d7f1a904ba
const ms = ThreeBodyMasses(m1=mp, m2=mπ, m3=mK, m0=mΛc)

# ╔═╡ d227ae5a-d779-4fa2-8483-8842b4f66563
function readjson(path)
    f = read(path, String)
    return JSON.parse(f)
end

# ╔═╡ 17c5b1c5-006f-4924-9e17-c2278985492c
const tbs = ThreeBodySystem(ms, ThreeBodySpins(1,0,0; two_h0=1))

# ╔═╡ c216abd3-f7c8-4fe6-8559-0ebbf04965cb
const parities = ['+', '-', '-', '±']

# ╔═╡ 65473482-5487-4d10-88a6-aedd62538014
md"""
## Isobar lineshapes:
`BreitWignerMinL`, `BuggBreitWignerMinL`, `Flatte1405`
"""

# ╔═╡ 8c091c8c-f550-4ea6-b4ad-11aa4027468c
breakup(m²,m1²,m2²) = sqrt(KallenFact(m²,m1²,m2²))/(2*sqrt(m²))

# ╔═╡ 9f936b78-9235-40ef-8e34-e6ce1d88734e
function F²(l,p,p0,d)
	pR = p*d
	p0R = p0*d
	l==0 && return 1.0
	l==1 && return (1+p0R^2)/(1+pR^2)
	l!=2 && error("l>2 cannot be")
	return (9+3p0R^2+p0R^4)/(9+3pR^2+pR^4)
end

# ╔═╡ fe3e1d61-1ad0-4509-92c7-dbac6a81343f
begin
	abstract type Lineshape end
	@with_kw struct BreitWignerMinL{T} <: Lineshape
		pars::T
		l::Int
		minL::Int
		# 
		name::String
		# 
		m1::Float64
		m2::Float64
		mk::Float64
		m0::Float64
	end
	BreitWignerMinL(pars::T; kw...) where T = BreitWignerMinL(; pars, kw...)
	function (BW::BreitWignerMinL)(σ)
		dR, dΛc = 1.5, 5.0 # /GeV
		m,Γ₀ = BW.pars
		@unpack l, minL = BW
		@unpack m1,m2,mk,m0 = BW
		p,p0 = breakup(σ,m1^2,m2^2), breakup(m^2,m1^2,m2^2)
		q,q0 = breakup(m0^2,σ,mk^2), breakup(m0^2,m^2,mk^2)
		Γ = Γ₀*(p/p0)^(2l+1)*m/sqrt(σ)*F²(l,p,p0,dR)
		1/(m^2-σ-1im*m*Γ) * (p/p0)^l * (q/q0)^minL *
			sqrt(F²(l,p,p0,dR) * F²(minL,q,q0,dΛc))
	end
	
	# BuggBreitWignerMinL
	@with_kw struct BuggBreitWignerMinL{T} <: Lineshape
		pars::T
		l::Int
		minL::Int
		# 
		name::String
		# 
		m1::Float64
		m2::Float64
		mk::Float64
		m0::Float64
	end
	BuggBreitWignerMinL(pars::T; kw...) where
		T <: NamedTuple{X, Tuple{Float64, Float64}} where X =
			BuggBreitWignerMinL(; pars=merge(pars, (γ=1.1,)), kw...)
	# 
	function (BW::BuggBreitWignerMinL)(σ)
		σA = mK^2-mπ^2/2
		m, Γ₀, γ = BW.pars
		@unpack m1,m2 = BW
		Γ = (σ-σA) / (m^2-σA) * Γ₀*exp(-γ*σ)# * breakup(σ,m1^2,m2^2)/(2*sqrt(σ))
		1/(m^2-σ-1im*m*Γ)
	end

	# Flatte1405
	@with_kw struct Flatte1405{T} <: Lineshape
		pars::T
		l::Int
		minL::Int
		#
		name::String
		# 
		m1::Float64
		m2::Float64
		mk::Float64
		m0::Float64
	end
	#
	Flatte1405(pars::T; kw...) where T = Flatte1405(; pars, kw...)
	function (BW::Flatte1405)(σ)
		m,Γ₀ = BW.pars
		@unpack m1, m2, m0, mk = BW
		p,p0 = breakup(σ,m1^2,m2^2), breakup(m^2,mπ^2,mΣ^2)
		p′,p0′ = breakup(σ,mπ^2,mΣ^2), breakup(m^2,mπ^2,mΣ^2)
		Γ1 = Γ₀*(p/p0)*m/sqrt(σ)
		Γ2 = Γ₀*(p′/p0′)*m/sqrt(σ)
		Γ = Γ1+Γ2
		1/(m^2-σ-1im*m*Γ)
	end
end

# ╔═╡ 68dbd288-7c0b-430a-9e6e-591532089c27
function updatepars(BW::Lineshape, pars)
	fiels = fieldnames(typeof(BW))
	values = [getproperty(BW,f) for f in fiels]
	return typeof(BW)(; NamedTuple{fiels}(values)..., pars)
end

# ╔═╡ cadbfb87-8774-428d-a2ef-337bd7465563
@recipe function f(BW::Lineshape)
	xv = range((BW.m1+BW.m2)^2, (BW.m0-BW.mk)^2, length=300)
	intensity(σ) = abs2(BW(σ)) * 
		breakup(σ, BW.m1^2, BW.m2^2) *
		breakup(BW.m0^2, σ, BW.mk^2) / sqrt(σ)
	yv = intensity.(xv)
	(xv, yv ./ sum(yv) .* length(yv))
end

# ╔═╡ 877f6aab-d0ff-44d7-9ce0-e43d895be297
md"""
## Parsing the isobar description
"""

# ╔═╡ 469a9d3e-6c82-4cd6-afb6-5e92c481a7a2
function ifhyphenaverage(s::String)
	factor = findfirst('-', s) === nothing ? 1 : 2
	eval(Meta.parse(replace(s,'-'=>'+'))) / factor
end

# ╔═╡ 680c04bb-027f-46ad-b53b-039e33dacd86
function buildchain(key, dict)
	@unpack mass, width, lineshape = dict
	#
	k = Dict('K'=>1,'D'=>3,'L'=>2)[first(key)]
	#
	jp_R = str2jp(dict["jp"])
	parity = jp_R.p
	two_j = jp_R.j |> x2
	#
	massval, widthval = ifhyphenaverage.((mass,width)) ./ 1e3
	#
	i,j = ij_from_k(k)
	# 
	@unpack two_js = tbs
	# 
	reaction_ij = jp_R => (jp(two_js[i]//2,parities[i]), jp(two_js[j]//2,parities[j]))
	reaction_Rk(P0) = jp(two_js[0]//2,P0) => (jp_R, jp(two_js[k]//2,parities[k]))
	# 
	LS = vcat(possible_ls.(reaction_Rk.(('+','-')))...)
	minLS = first(sort(vcat(LS...); by=x->x[1]))
	# 
	ls = possible_ls(reaction_ij)
	length(ls) != 1 && error("expected the only ls: $(ls)")
	onlyls = first(ls)
	#
	Hij = ParityRecoupling(two_js[i],two_js[j],reaction_ij)
	Xlineshape = eval(
		quote 
		$(Symbol(lineshape))(
				(; m=$massval, Γ=$widthval);
				name = $key,
				l = $(onlyls[1]),
				minL = $(minLS[1]),
				m1=$(ms[i]), m2=$(ms[j]), mk=$(ms[k]), m0=$(ms[0]))
		end)
	return (; k, Xlineshape, Hij, two_s=two_j, parity)
end

# ╔═╡ 7cc4c5f9-4392-4b57-88af-59d1cf308162
isobarsinput = readjson(joinpath("..","data","isobars.json"))["isobars"];

# ╔═╡ f9449a5b-7d92-43d7-95a6-c9e4632172f5
typeof(
	[buildchain("K(892)", isobarsinput["K(892)"]).Xlineshape,
	 buildchain("L(1405)", isobarsinput["L(1405)"]).Xlineshape])

# ╔═╡ b0f5c181-dcb2-48f8-a510-57eac44ca4d9
begin
	isobars = Dict()
	#
	for (key, dict) in isobarsinput
		isobars[key] = buildchain(key, dict)
	end
	isobars
end ;

# ╔═╡ c71bd0a0-80db-473a-b7b4-8018a8d86580
 let
 	BW = isobars["K(700)"].Xlineshape
	@unpack m1,m2 = BW
	m,Γ = BW.pars
 	breakup(m^2,m1^2,m2^2)/(2*sqrt(m^2))
 end

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

# ╔═╡ f8d9463f-2150-4631-84c1-9afdb32626be
let
	plot(layout=grid(1,length(parameterupdates)), size=(700,200))
	for (sp,(p,u)) in enumerate(parameterupdates)
		BW = isobars[p].Xlineshape
		plot!(BW; sp, lab=p)
		plot!(updatepars(BW, merge(BW.pars,u)); sp)
	end
	plot!()
end

# ╔═╡ 1f6dc202-8734-4232-9f48-a88ebf17ff93
for (p,u) in parameterupdates
	BW = isobars[p].Xlineshape
	isobars[p] = merge(isobars[p], (Xlineshape=updatepars(BW, merge(BW.pars,u)),))
end

# ╔═╡ 21c125fc-e218-4676-9846-822d951f4f1b
md"""
### Coupligns
"""

# ╔═╡ 58800d91-9cd9-4a9e-95a9-6b285157385d
function selectindexmap(isobarname)
	# 
	couplingindexmap = Dict(
		r"[L|D].*" => Dict(
				'1' => (1, 0),
				'2' => (-1, 0)),
		r"K\(892\)" => Dict(
				'1' => (0,  1),
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

# ╔═╡ 5f8973a8-faa7-4663-b896-e7f4ff9600a2
function couplingLHCb2DPD(two_λR, two_λk; k, parity, two_j)
	if k == 2
		@assert two_λk == 0
		c′ = - (2*(parity == '+')-1) / sqrt(two_j+1)
		return (-two_λR, two_λk, c′)
	elseif k == 3
		c′ = - (2*(parity == '+')-1) * minusone()^(two_j//2-1//2) / sqrt(two_j+1)
		return (-two_λR, two_λk, c′)
	end
	k!=1 && error("cannot be!")
	c′ = 1.0 / sqrt(two_j+1)
	return (two_λR, -two_λk, c′)
end

# ╔═╡ e9610bef-edca-43ac-96af-ad118c6879c7
"""
The relation is
```math
A^{DPD}_{λ₀,λ₁} = (-1)^{½-λ₁} A^{LHCb}_{λ₀,-λ₁}
```
"""
amplitudeLHCb2DPD(A) =
	[A[1,2] -A[1,1]
	 A[2,2] -A[2,1]]

# ╔═╡ 64a1d1f5-da82-4ce6-8731-ccc7439bd881
couplingkeys = collect(filter(x->x[1:2]=="Ar", keys(defaultparameters)))

# ╔═╡ 78dba088-6d5b-4e4b-a664-f176f9e2d673
isobarnames = map(x->x[3:end-1], couplingkeys)

# ╔═╡ 28aceaed-9143-4b26-89d7-6763b2fdbc28
function parname2decaychain(parname)
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

# ╔═╡ 168707da-d42c-4b2f-94b6-f7bc15cb29cb
const terms = [let
	# 
	c_re_key = "Ar" * parname[3:end] # = parname
	c_im_key = "Ai" * parname[3:end]
	value_re = eval(Meta.parse(defaultparameters[c_re_key])).val
	value_im = eval(Meta.parse(defaultparameters[c_im_key])).val
	value = value_re + 1im*value_im
	#
	c0, d = parname2decaychain(parname)
	# 
	(c0*value, d)
end for parname in couplingkeys]

# ╔═╡ 8d8824e8-7809-4368-840c-b8f6d38ad7c2
begin
	const chains = getindex.(terms,2)
	const couplings = getindex.(terms,1)
end;

# ╔═╡ 40a85984-0588-4e72-bb99-ab555d82c020
md"""
### Cross check with the LHCb code
"""

# ╔═╡ 38234e92-aa91-4728-b8d8-8f7dd9f25552
crosscheckresult = readjson(joinpath("..","data","crosscheck.json")) ;

# ╔═╡ 0867cb8e-58b4-4d23-9b1a-e18063848252
parsepythoncomplex(s::String) = eval(Meta.parse(
	replace(s,
			"("=>"",
			")"=>"",
		"j"=>"im")))

# ╔═╡ 96b0a235-e4cf-406d-a72b-124fec4e6ba7
Adict2matrix(d::Dict) = parsepythoncomplex.(
	[d["A++"] d["A+-"]
	 d["A-+"] d["A--"]])

# ╔═╡ 210740f1-769a-456c-b075-7fab2728bda6
σs0 = Invariants(ms,
	σ1 = 0.7980703453578917,
	σ2 = 3.6486261122281745)

# ╔═╡ f2570431-067f-424c-822d-9c7a8d6764f1
begin
	myK892BW0 = parname2decaychain("ArK(892)1")[2].Xlineshape(σs0[1])
	tfK892BW0 = 2.1687201455088894+23.58225917009096im
	myK892BW0 ≈ tfK892BW0
end

# ╔═╡ e972aeb7-2724-4f7b-b00e-744e2ee7ef8f
begin
	myL1405BW0 = parname2decaychain("ArL(1405)1")[2].Xlineshape(σs0[2])
	tfL1405BW0 = -0.5636481410171861+0.13763637759224928im
	tfL1405BW0 ≈ myL1405BW0
end

# ╔═╡ bb8588a5-b439-4dc5-a15c-da894c20fbb3
begin
	comparison = DataFrame()
	for parname in couplingkeys
		c, d = parname2decaychain(parname)
		M_DPD = [c * amplitude(σs0,[two_λ1,0,0,two_λ0], d)
			for (two_λ0,two_λ1) in [(1, 1) ( 1,-1)
								    (-1,1) (-1,-1)]]
		chainamps = crosscheckresult["chains"]
		M_LHCb′ = amplitudeLHCb2DPD(Adict2matrix(chainamps[parname]))
		#
		r = filter(x->!(isnan(x)), vcat(M_DPD ./ M_LHCb′))
		push!(comparison, (; parname=parname[3:end], r, M_DPD, M_LHCb′))
	end
	sort!(comparison, order(:parname, by=x->x[3:6]))
	sort!(comparison,
		order(:parname, by=x->findfirst(x[1],"LDK")))
end

# ╔═╡ 849cf226-f85f-4672-8e10-30bb0de4ad93
extrema(real(vcat(comparison.r...))) .- 1, 
extrema(imag(vcat(comparison.r...)))

# ╔═╡ c39e1b06-642a-4f8f-a01a-814d00f98977
isobars["K(1430)"].Xlineshape.pars

# ╔═╡ 7c09ae7b-f57d-44c3-a69b-aaf9ebd129fd
select(
	transform(
		transform(comparison,
			:parname => ByRow(x->isobars[x[1:end-1]].Xlineshape.l) => :l),	
	:parname => ByRow(x->isobars[x[1:end-1]].Xlineshape.minL) => :Lmin),
[:parname,:r,:l,:Lmin])

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

# ╔═╡ f31d7e02-865d-411d-8fdf-e9ff4753b139
intensity(Ai::AbstractArray, ci::Vector) = sum(abs2, sum(a .* ci) for a in Ai)

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

# ╔═╡ c945a6b3-e597-4409-840b-d806cc2958ad


# ╔═╡ 4de4f76f-e0b9-4f1e-a166-8c0ae5397334
two_Δλ(HRk) = HRk.two_λa-HRk.two_λb

# ╔═╡ 0a976167-b074-4694-ab97-aecfcd67cc25
begin
	rates = DataFrame()
	for s in Set(isobarnames)
		couplingsmap = (isobarnames .== s)
		Iξv = intensity.(Aiv, Ref(couplings .* couplingsmap));
		# 
		weights = two_Δλ.(getproperty.(chains[couplingsmap], :HRk))
		α = sum(abs2.(couplings[couplingsmap]) .* weights) / 
			sum(abs2, couplings[couplingsmap])
		#
		push!(rates, (isobarname=s, rate = sum(Iξv) / I0 * 100, α))
	end
	sort(
		sort(
			transform(rates, :rate=>ByRow(x->round(x; digits=2)); renamecols=false),
			order(:isobarname, by=x->eval(Meta.parse(x[3:end-1])))),
		order(:isobarname, by=x->findfirst(x[1],"LDK")))
end

# ╔═╡ cecc3975-ede0-4f00-8259-8e2b3e69022c
begin
	# 
	delta_i(n,iv...) = (v=zeros(n); v[[iv...]] .= 1.0; v)
	nchains = length(couplings)
	ratematrix = zeros(nchains,nchains)
	for i in 1:nchains, j in i:nchains
		cij = delta_i(nchains,i,j) .* couplings
		Iξv = intensity.(Aiv, Ref(cij))
		ratematrix[i,j] = sum(Iξv) / I0 * 100
		ratematrix[j,i] = ratematrix[i,j]
	end
	for i in 1:nchains, j in i+1:nchains
		ratematrix[i,j] += -ratematrix[i,i]-ratematrix[j,j]
		ratematrix[j,i] += -ratematrix[i,i]-ratematrix[j,j]
		ratematrix[i,j] /= 2
		ratematrix[j,i] /= 2
	end
end

# ╔═╡ d8477cb9-30c1-4aa5-808f-d90ec520b51f
@assert sum(ratematrix) ≈ 100

# ╔═╡ 64cdff72-fb69-413c-b4f3-d5629969c7b8
grouppedchains = getindex.(
	sort(sort(collect(enumerate(isobarnames)),
		by=x->x[2]),
		by=x->findfirst(x[2][1],"LDK")),1) ;

# ╔═╡ c1387ca0-59c6-44c8-99b3-bb53e44d638e
let
	grouppedratematrix = ratematrix[grouppedchains,grouppedchains]
	# 
	s(two_λ) = iseven(two_λ) ?
		string(div(two_λ,2)) :
		("-","")[1+div((sign(two_λ)+1),2)]*"½"
	labelchain(chain) = chain.Xlineshape.name * " "*
		s(chain.HRk.two_λa)*","*s(chain.HRk.two_λb)
	labels = labelchain.(chains)
	# 
	clim = maximum(ratematrix) .* (-1, 1)
	heatmap(grouppedratematrix;
		xticks=(1:nchains, labels), xrotation = 90,
		yticks=(1:nchains, labels), aspectratio=1,
		size=(600,600), c=:delta, colorbar=true,
		title="Rate matrix for chains", clim)
	for i in 1:nchains, j in i+1:nchains
		annotate!((i,j,text(
			grouppedratematrix[i,j] ≥ 0 ? "+" : "-",4)))
	end
	plot!()
end

# ╔═╡ 2faffd58-3fb8-4afe-ad04-b2c84afc0d60
group(ratematrix, sectors) =
	[sum(getindex(ratematrix, iv, jv))
	for iv in sectors,
		jv in sectors]

# ╔═╡ a9e20455-2180-4e63-a0af-bab800fa0616
let
	isobarnameset = collect(Set(isobarnames))
	sort!(isobarnameset, by=x->findfirst(x[1],"LDK"))
	# 
	sectors = [couplingsmap = collect(1:nchains)[(isobarnames .== s)]
		for s in isobarnameset]
	# 
	grouppedratematrix = group(ratematrix, sectors)
	nisobars = length(isobarnameset)
	for i in 1:nisobars, j in i+1:nisobars
		grouppedratematrix[j,i] *= 2
		grouppedratematrix[i,j] = 0
	end
	@assert sum(grouppedratematrix) ≈ 100
	
	clim = maximum(grouppedratematrix) .* (-1.2, 1.2)
	heatmap(grouppedratematrix;
		xticks=(1:nchains, isobarnameset), xrotation = 90,
		yticks=(1:nchains, isobarnameset), aspectratio=1,
		size=(600,600), c=:delta, colorbar=true,
		title="Rate matrix for isobars", clim)
	for i in 1:nisobars, j in i:nisobars
		annotate!((i,j,text(
			round(grouppedratematrix[j,i],digits=2), 6,
			i==j ? :red : :black)))
	end
	plot!()
end

# ╔═╡ Cell order:
# ╟─d3cb114e-ee0c-4cd5-87fb-82289849aceb
# ╠═dea88954-c7b2-11ec-1a3d-717739cfd08b
# ╠═e04157cc-7697-41e5-8e4e-3556332929ef
# ╠═24d2ac10-b474-472d-827b-c353f44d9e8c
# ╠═cfe26457-a531-4fc4-8eee-34d7f1a904ba
# ╠═d227ae5a-d779-4fa2-8483-8842b4f66563
# ╠═17c5b1c5-006f-4924-9e17-c2278985492c
# ╠═c216abd3-f7c8-4fe6-8559-0ebbf04965cb
# ╟─65473482-5487-4d10-88a6-aedd62538014
# ╠═8c091c8c-f550-4ea6-b4ad-11aa4027468c
# ╠═9f936b78-9235-40ef-8e34-e6ce1d88734e
# ╠═fe3e1d61-1ad0-4509-92c7-dbac6a81343f
# ╠═c71bd0a0-80db-473a-b7b4-8018a8d86580
# ╠═68dbd288-7c0b-430a-9e6e-591532089c27
# ╠═cadbfb87-8774-428d-a2ef-337bd7465563
# ╟─877f6aab-d0ff-44d7-9ce0-e43d895be297
# ╠═469a9d3e-6c82-4cd6-afb6-5e92c481a7a2
# ╠═680c04bb-027f-46ad-b53b-039e33dacd86
# ╠═7cc4c5f9-4392-4b57-88af-59d1cf308162
# ╠═f9449a5b-7d92-43d7-95a6-c9e4632172f5
# ╠═b0f5c181-dcb2-48f8-a510-57eac44ca4d9
# ╟─98cca824-73db-4e1c-95ae-68c3bc8574fe
# ╟─c7572ffb-c4c7-4ce6-90a9-8237214ac91b
# ╠═cee9dc28-8048-49e7-8caf-8e07bcd884c4
# ╟─30c3c8ef-ad69-43e6-9a75-525dfbf7007a
# ╠═7ecc65a4-9fe7-4209-ad54-f1c8abe52ee5
# ╠═24d051e5-4d9e-48c8-aace-225f3a0218cb
# ╠═f9158b4d-4d27-4aba-bf5a-529135ec48e2
# ╠═1db41980-ea36-4238-90cd-bf2427772ea9
# ╠═f8d9463f-2150-4631-84c1-9afdb32626be
# ╠═1f6dc202-8734-4232-9f48-a88ebf17ff93
# ╟─21c125fc-e218-4676-9846-822d951f4f1b
# ╠═58800d91-9cd9-4a9e-95a9-6b285157385d
# ╠═5f8973a8-faa7-4663-b896-e7f4ff9600a2
# ╟─e9610bef-edca-43ac-96af-ad118c6879c7
# ╠═64a1d1f5-da82-4ce6-8731-ccc7439bd881
# ╠═78dba088-6d5b-4e4b-a664-f176f9e2d673
# ╠═28aceaed-9143-4b26-89d7-6763b2fdbc28
# ╠═168707da-d42c-4b2f-94b6-f7bc15cb29cb
# ╠═8d8824e8-7809-4368-840c-b8f6d38ad7c2
# ╟─40a85984-0588-4e72-bb99-ab555d82c020
# ╠═f2570431-067f-424c-822d-9c7a8d6764f1
# ╠═e972aeb7-2724-4f7b-b00e-744e2ee7ef8f
# ╠═38234e92-aa91-4728-b8d8-8f7dd9f25552
# ╠═0867cb8e-58b4-4d23-9b1a-e18063848252
# ╠═96b0a235-e4cf-406d-a72b-124fec4e6ba7
# ╠═210740f1-769a-456c-b075-7fab2728bda6
# ╠═bb8588a5-b439-4dc5-a15c-da894c20fbb3
# ╠═849cf226-f85f-4672-8e10-30bb0de4ad93
# ╠═c39e1b06-642a-4f8f-a01a-814d00f98977
# ╠═7c09ae7b-f57d-44c3-a69b-aaf9ebd129fd
# ╟─d547fc89-3756-49d3-8b49-ad0c56c5c3a3
# ╠═abdeb1ac-19dc-45b2-94dd-5e64fb3d8f14
# ╠═e50afc73-d0a6-4a8d-ae06-c95c9946998d
# ╠═b833843f-c8b6-44c6-847c-efce0ed989d4
# ╟─b27001c0-df6c-4a47-ae53-8cee96cbf984
# ╠═0736dd22-89dd-4d7f-b332-b0767180ad43
# ╠═f31d7e02-865d-411d-8fdf-e9ff4753b139
# ╠═88c58ce2-c98b-4b60-901f-ed95099c144b
# ╠═589c3c6f-fc24-4c2c-bf42-df0b3187d8cf
# ╠═b3c0d027-4c98-47e4-9f9a-77ba1d10ae98
# ╠═50d4a601-3f14-4034-9e6f-08eae9ca7d7c
# ╠═8e06d5ec-4b98-441d-919b-7b90071e6674
# ╠═fc62283e-8bcb-4fd1-8809-b7abeb991030
# ╠═c945a6b3-e597-4409-840b-d806cc2958ad
# ╠═4de4f76f-e0b9-4f1e-a166-8c0ae5397334
# ╠═0a976167-b074-4694-ab97-aecfcd67cc25
# ╠═cecc3975-ede0-4f00-8259-8e2b3e69022c
# ╠═d8477cb9-30c1-4aa5-808f-d90ec520b51f
# ╠═64cdff72-fb69-413c-b4f3-d5629969c7b8
# ╠═c1387ca0-59c6-44c8-99b3-bb53e44d638e
# ╠═2faffd58-3fb8-4afe-ad04-b2c84afc0d60
# ╠═a9e20455-2180-4e63-a0af-bab800fa0616

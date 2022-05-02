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
	using RecipesBase
	# 
	using LinearAlgebra
	using Parameters
	using Measurements
	using DataFrames
	# 
	using ThreeBodyDecay
	using ThreeBodyDecay.PartialWaveFunctions
end

# ╔═╡ e04157cc-7697-41e5-8e4e-3556332929ef
theme(:wong2, frame=:box, grid=false, minorticks=true, 
    guidefontvalign=:top, guidefonthalign=:right,
    xlim=(:auto,:auto), ylim=(:auto,:auto),
    lw=1, lab="", colorbar=false)

# ╔═╡ 24d2ac10-b474-472d-827b-c353f44d9e8c
begin
	const mK = 0.493
	const mπ = 0.140
	const mp = 0.938
	const mΣ = 1.18925
	const mΛc = 2.28
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
## Lineshapes
"""

# ╔═╡ 8c091c8c-f550-4ea6-b4ad-11aa4027468c
breakup(m²,m1²,m2²) = sqrt(KallenFact(m²,m1²,m2²))

# ╔═╡ 01217062-003d-4496-b0fe-f54d32fbb2bf
md"""
### `BreitWignerMinL`
"""

# ╔═╡ 9f936b78-9235-40ef-8e34-e6ce1d88734e
function F²(l,p,p0,d)
	pR = p*d
	p0R = p0*d
	l==0 && return 1.0
	l==1 && return (1+p0R^2)/(1+pR^2)
	l!=2 && error("l>2 cannot be")
	return (9+3p0R^2+3p0R^4)/(9+3pR^2+3pR^4)
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
		mΓ = (σ-σA) / (m^2-σA) * Γ₀^2*exp(-γ*σ)
		1/(m^2-σ-1im*mΓ)
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
		Γ1 = Γ₀*(p/p0)*m/sqrt(σ)  # should also have F
		Γ2 = Γ₀*(p′/p0′)*m/sqrt(σ)  # should also have F
		Γ = Γ1+Γ2
		1/(m^2-σ-1im*m*Γ)
	end
end

# ╔═╡ 877f6aab-d0ff-44d7-9ce0-e43d895be297
md"""
## Parsing the input
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
	reaction_ij = jp_R=>(jp(two_js[i]//2,parities[i]), jp(two_js[j]//2,parities[j]))
	reaction_Rk(P0) = jp(two_js[0]//2,P0) => (jp_R, jp(two_js[k]//2,parities[k]))
	# 
	LS = vcat(possible_ls(reaction_Rk('+')), possible_ls(reaction_Rk('-')))
	minLS = first(sort(vcat(LS...); by=x->x[1]))
	# 
	ls = possible_ls(reaction_ij)
	length(ls) != 1 && error("expected the only on ls: $(ls)")
	onlyls = first(ls)
	#
	Hij = ParityRecoupling(two_js[i],two_js[j],reaction_ij)
	# HRk = RecouplingLS(minLS |> x2, reaction_Rk('+'))
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
end

# ╔═╡ cee9dc28-8048-49e7-8caf-8e07bcd884c4
let
	plot(layout=grid(1,3), size=(1000,240))
	for (k,v) in isobars
		xv = range(lims(v.k,ms)..., length=300)
		calv = abs2.(v.Xlineshape.(xv))
		plot!(sp=v.k, xv, calv ./ sum(calv), lab=k)
	end
	plot!()
end

# ╔═╡ 3971792d-a41f-4792-92b7-8b36e51f70b1
# plot([
# 	plot(ms,
# 	summed_over_polarization((x,y)->abs2(amplitude(x,y,d)), tbs.two_js)
# 	; iσx=1, iσy=2, title=d.Xlineshape.name) for d in ds]...)

# ╔═╡ 7ecc65a4-9fe7-4209-ad54-f1c8abe52ee5
 modelparameters =
	 readjson(joinpath("..","data","modelparameters.json"))["modelstudies"]

# ╔═╡ 24d051e5-4d9e-48c8-aace-225f3a0218cb
begin
	defaultparameters = first(modelparameters)["parameters"]
	defaultparameters["ArK(892)1"] = "1.0 ± 0.0"
	defaultparameters["AiK(892)1"] = "0.0 ± 0.0"
end

# ╔═╡ 468884de-530f-4903-acfa-72946e330866
couplingindexmap = Dict(
	r"L.*" => Dict(
			'1' => (1, 0),
			'2' => (-1, 0)),
	r"D.*" => Dict(
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

# ╔═╡ f3e20fe4-1e26-47e0-9a2d-a794ece9e0d9
begin
	modelterms = DataFrame()
	# 
	couplingkeys = filter(x->x[1:2]=="Ar", keys(defaultparameters))
	for c in couplingkeys
		isobarname = c[3:end-1]
		couplingindex = c[end]
		c_re = c
		c_im = "Ai" * c[3:end]
		value_re = eval(Meta.parse(defaultparameters[c_re])).val
		value_im = eval(Meta.parse(defaultparameters[c_im])).val
		value_re + 1im*value_im
		# 
		r = filter(keys(couplingindexmap)) do k
			match(k, isobarname) !== nothing
		end
		(two_λR,two_λk) = couplingindexmap[first(r)][couplingindex]
		# 
		push!(modelterms, (; isobarname, two_λR,two_λk, c=value_re + 1im*value_im, ))
	end
	modelterms
end

# ╔═╡ 5f8973a8-faa7-4663-b896-e7f4ff9600a2
function convertLHCb2DPD(two_λR, two_λk, c; k, parity, two_j)
	if k == 2
		@assert two_λk == 0
		c′ = - (2*(parity == '+')-1) * c
		return (-two_λR, two_λk, c′)
	elseif k == 3
		c′ = - (2*(parity == '+')-1) * minusone()^(two_j//2-1//2) * c
		return (-two_λR, two_λk, c′)
	end
	k!=1 && error("cannot be!")
	return (two_λR, -two_λk, c)
end

# ╔═╡ fe72915f-51c7-48bd-8745-9bc97d05162c
const terms = [
	begin
		@unpack k, Hij, two_s, Xlineshape, parity = isobars[t.isobarname]
		two_λR, two_λk, c =
			convertLHCb2DPD(t.two_λR, t.two_λk, t.c; k, two_j=two_s, parity)
		HRk = NoRecoupling(two_λR, two_λk)
		(c, DecayChain(; k, Xlineshape, Hij, HRk, two_s, tbs))
	end for t in eachrow(modelterms)]

# ╔═╡ abdeb1ac-19dc-45b2-94dd-5e64fb3d8f14
A(σs,two_λs) = sum(c*amplitude(σs,two_λs, d) for (c,d) in terms)

# ╔═╡ e50afc73-d0a6-4a8d-ae06-c95c9946998d
begin
	I = summed_over_polarization(abs2 ∘ A, tbs.two_js)
end

# ╔═╡ 99be3b6e-4462-4bf9-bf04-5c4e61966e28
begin
	λσs = randomPoint(tbs)
	I(λσs.σs)
end

# ╔═╡ 88c58ce2-c98b-4b60-901f-ed95099c144b
pdata = flatDalitzPlotSample(ms; Nev=10000)

# ╔═╡ b833843f-c8b6-44c6-847c-efce0ed989d4
plot(ms,σs->I(σs), iσx=2, iσy=1)

# ╔═╡ e2346d61-a4c7-40bf-9e0d-87948da4e160
Iv = I.(pdata)

# ╔═╡ 8e06d5ec-4b98-441d-919b-7b90071e6674
histogram2d(
	getproperty.(pdata, :σ2),
	getproperty.(pdata, :σ1), weights=Iv, bins=100)

# ╔═╡ Cell order:
# ╠═dea88954-c7b2-11ec-1a3d-717739cfd08b
# ╠═e04157cc-7697-41e5-8e4e-3556332929ef
# ╠═24d2ac10-b474-472d-827b-c353f44d9e8c
# ╠═cfe26457-a531-4fc4-8eee-34d7f1a904ba
# ╠═d227ae5a-d779-4fa2-8483-8842b4f66563
# ╠═17c5b1c5-006f-4924-9e17-c2278985492c
# ╠═c216abd3-f7c8-4fe6-8559-0ebbf04965cb
# ╟─65473482-5487-4d10-88a6-aedd62538014
# ╠═8c091c8c-f550-4ea6-b4ad-11aa4027468c
# ╟─01217062-003d-4496-b0fe-f54d32fbb2bf
# ╠═9f936b78-9235-40ef-8e34-e6ce1d88734e
# ╠═fe3e1d61-1ad0-4509-92c7-dbac6a81343f
# ╟─877f6aab-d0ff-44d7-9ce0-e43d895be297
# ╠═469a9d3e-6c82-4cd6-afb6-5e92c481a7a2
# ╠═680c04bb-027f-46ad-b53b-039e33dacd86
# ╠═7cc4c5f9-4392-4b57-88af-59d1cf308162
# ╠═f9449a5b-7d92-43d7-95a6-c9e4632172f5
# ╠═b0f5c181-dcb2-48f8-a510-57eac44ca4d9
# ╠═cee9dc28-8048-49e7-8caf-8e07bcd884c4
# ╠═3971792d-a41f-4792-92b7-8b36e51f70b1
# ╠═7ecc65a4-9fe7-4209-ad54-f1c8abe52ee5
# ╠═24d051e5-4d9e-48c8-aace-225f3a0218cb
# ╠═468884de-530f-4903-acfa-72946e330866
# ╠═f3e20fe4-1e26-47e0-9a2d-a794ece9e0d9
# ╠═5f8973a8-faa7-4663-b896-e7f4ff9600a2
# ╠═fe72915f-51c7-48bd-8745-9bc97d05162c
# ╠═abdeb1ac-19dc-45b2-94dd-5e64fb3d8f14
# ╠═e50afc73-d0a6-4a8d-ae06-c95c9946998d
# ╠═99be3b6e-4462-4bf9-bf04-5c4e61966e28
# ╠═88c58ce2-c98b-4b60-901f-ed95099c144b
# ╠═b833843f-c8b6-44c6-847c-efce0ed989d4
# ╠═e2346d61-a4c7-40bf-9e0d-87948da4e160
# ╠═8e06d5ec-4b98-441d-919b-7b90071e6674

### A Pluto.jl notebook ###
# v0.19.3

# cspell:ignore onlyls Xlineshape

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
breakup(m²,m1²,m2²) = sqrt(Kallen(m²,m1²,m2²))

# ╔═╡ 01217062-003d-4496-b0fe-f54d32fbb2bf
md"""
### `BreitWignerMinL`
"""

# ╔═╡ 7a5a7e49-5454-4021-a698-766b1242f757
begin
	@with_kw struct BreitWignerMinL{T}
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
	function (BW::BreitWignerMinL)(s,σ)
		m,Γ₀ = BW.pars
		@unpack l, minL = BW
		@unpack m1,m2,mk,m0 = BW
		p,p0 = breakup(σ,m1^2,m2^2), breakup(m^2,m1^2,m2^2)
		q,q0 = breakup(s,σ,mk^2), breakup(s,m^2,mk^2)
		Γ = Γ₀*(p/p0)^(2l+1)*m/sqrt(σ)  # should also have F
		1/(m^2-σ-1im*m*Γ) * (p/p0)^l * (q/q0)^minL # should also have F
	end
end

# ╔═╡ 60a8cb20-6041-494d-9523-397ce329c7b6
md"""
### `BuggBreitWignerMinL`
"""

# ╔═╡ 94c6394f-d00c-462c-ae29-ea107b66e191
begin
	@with_kw struct BuggBreitWignerMinL{T}
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
	function (BW::BuggBreitWignerMinL)(s,σ)
		σA = mK^2-mπ^2/2
		m, Γ₀, γ = BW.pars
		mΓ = (σ-σA) / (m^2-σA) * Γ₀^2*exp(-γ*σ)
		1/(m^2-σ-1im*mΓ)
	end
end

# ╔═╡ 8baf8a30-fda8-4094-bcfd-4f48c71dc1d1
md"""
### `Flatte1405`
"""

# ╔═╡ 6b9a1178-7863-4493-a4fa-78676cc2695c
begin
	@with_kw struct Flatte1405{T}
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
	function (BW::Flatte1405)(s,σ)
		m,Γ₀ = BW.pars
		@unpack m1, m2 = BW
		p,p0 = breakup(σ,m1^2,m2^2), breakup(m^2,m1^2,m2^2)
		p′,p0′ = breakup(σ,mπ^2,mΣ^2), breakup(m^2,mp^2,mΣ^2)
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
	parity = dict["jp"][end]
	two_j = Int(2*eval(Meta.parse(dict["jp"][1:end-2])))
	#
	massval, widthval = ifhyphenaverage.((mass,width)) ./ 1e3
	#
	LS = vcat(
		[possible_ls(
			jp(two_j // 2, parity),
			jp(tbs.two_js[k] // 2, parities[k]);
			jp=jp(tbs.two_js[0] // 2, P0)) for P0 in ['+', '-']]...)
	#
	i,j = ij_from_k(k)
	ls = possible_ls(
		jp(tbs.two_js[i] // 2, parities[i]),
		jp(tbs.two_js[j] // 2, parities[j]);
		jp=jp(two_j // 2, parity))
	length(ls) != 1 && error("weird")
	onlyls = first(ls)
	minLS = first(sort(vcat(LS...); by=x->x[1]))
	#
	code = quote
		decay_chain(
			k=$k,
			Xlineshape = $(Symbol(lineshape))(
				(; m=$massval, Γ=$widthval);
				name = $key,
				l = $(onlyls[1]),
				minL = $(minLS[1]),
				m1=$(ms[i]), m2=$(ms[j]), mk=$(ms[k]), m0=$(ms[0])),
			two_LS = $(minLS .|> x2),
			two_ls = $(onlyls .|> x2),
			two_s=$two_j, tbs=$tbs,
			)
	end
	eval(code)
	# code
end

# ╔═╡ c034441c-95b0-409f-a3cf-2714789b2d0f
isobarsinput = readjson(joinpath("..","data","isobars.json"))["isobars"];

# ╔═╡ b0f5c181-dcb2-48f8-a510-57eac44ca4d9
ds = [buildchain(keys, dict) for (keys, dict) in isobarsinput]

# ╔═╡ cee9dc28-8048-49e7-8caf-8e07bcd884c4
let
	plot(layout=grid(1,3), size=(1000,240))
	for d in ds
		plot!(sp=d.k, x->abs2(d.Xlineshape(ms[0]^2,x)),lims(d.k,ms)...,
			lab=d.Xlineshape.name)
	end
	plot!()
end

# ╔═╡ 3971792d-a41f-4792-92b7-8b36e51f70b1
# plot([
# 	plot(ms,
# 	summed_over_polarization((x,y)->abs2(amplitude(x,y,d)), tbs.two_js)
# 	; iσx=1, iσy=2, title=d.Xlineshape.name) for d in ds]...)

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
# ╠═7a5a7e49-5454-4021-a698-766b1242f757
# ╟─60a8cb20-6041-494d-9523-397ce329c7b6
# ╠═94c6394f-d00c-462c-ae29-ea107b66e191
# ╟─8baf8a30-fda8-4094-bcfd-4f48c71dc1d1
# ╠═6b9a1178-7863-4493-a4fa-78676cc2695c
# ╟─877f6aab-d0ff-44d7-9ce0-e43d895be297
# ╠═469a9d3e-6c82-4cd6-afb6-5e92c481a7a2
# ╠═680c04bb-027f-46ad-b53b-039e33dacd86
# ╠═c034441c-95b0-409f-a3cf-2714789b2d0f
# ╠═b0f5c181-dcb2-48f8-a510-57eac44ca4d9
# ╠═cee9dc28-8048-49e7-8caf-8e07bcd884c4
# ╠═3971792d-a41f-4792-92b7-8b36e51f70b1

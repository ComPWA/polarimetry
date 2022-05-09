### A Pluto.jl notebook ###
# v0.18.0

using Markdown
using InteractiveUtils

# ╔═╡ b56fc9a7-fa03-47d6-b22e-0097be7155d3
begin
	cd(joinpath(@__DIR__,".."))
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	#
	using Plots
	using RecipesBase
	#
	using LinearAlgebra
	using Parameters
	#
	using ThreeBodyDecay
end

# ╔═╡ 4f00e2b5-6097-4577-b669-eafe4cbf1852
using PlutoUI

# ╔═╡ d5d6b8bf-1321-464d-80bb-be1cc1b1ed8d
with_terminal() do
	Pkg.status()
end

# ╔═╡ d65ed080-9654-11ec-0a8e-e530b9164a8c
Rz(p,θ) = [p[1]*cos(θ)-p[2]*sin(θ), p[2]*cos(θ)+p[1]*sin(θ), p[3]]

# ╔═╡ 9beb287e-91ee-4a01-b803-5686576cf087
Ry(p,cθ,sθ) = [p[1]*cθ+p[3]*sθ, p[2], p[3]*cθ-p[1]*sθ]

# ╔═╡ 55cf2a64-dd00-461a-ab93-cad2ef71fbb0
Ry(p,θ) = [p[1]*cos(θ)+p[3]*sin(θ), p[2], p[3]*cos(θ)-p[1]*sin(θ)]

# ╔═╡ f341c6fd-4b7a-4497-acf8-f343a95e9521
begin
	Rz(θ) = x->Rz(x,θ)
	Ry(θ) = x->Ry(x,θ)
end

# ╔═╡ 4a5e2b92-37fc-494b-8c00-d8a339ce7bcb
const ms = ThreeBodyMasses(m1=0.938272046, m2=0.13957018, m3=0.493677, m0=2.28646)

# ╔═╡ 570f8ee0-7c75-4c93-bc3a-debab0b81874
const ms² = ms^2

# ╔═╡ 1a0a3c20-0990-4a7d-8395-941bf34ac9e3
function constructinvariants(σs)
	p₁ = sqrt(Kallen(ms.m0^2,ms[1]^2,σs[1])) / (2ms.m0)
	p₂ = sqrt(Kallen(ms.m0^2,ms[2]^2,σs[2])) / (2ms.m0)
	p₃ = sqrt(Kallen(ms.m0^2,ms[3]^2,σs[3])) / (2ms.m0)
	#
	p1 = [0,0,p₁]
	p2 = (c=cosζ(wr(1,2,0),σs,ms²); Ry([0,0,p₂], c, -sqrt(1-c^2)))
	p3 = (c=cosζ(wr(3,1,0),σs,ms²); Ry([0,0,p₃], c, sqrt(1-c^2)))
	return (; p1, p2, p3)
end

# ╔═╡ 34de7c63-83db-4247-b904-1eaa5822af18
md"""
### LHCb-PAPER-2022-002 relations
"""

# ╔═╡ 9a793648-ae8f-465d-9a55-0735a16fd1db
function alteulerangles(p1′,p2′,p3′)
	ϕp = atan(p1′[2],p1′[1])
	θp = acos(p1′[3]/norm(p1′))
	χ = atan((p1′ × p3′)[3], -(p1′ × p3′ × p1′)[3] / norm(p1′))
	#
	(; ϕp, θp, χ)
end

# ╔═╡ 89d46224-82a3-4b53-b17e-4fb7ba702202
md"""
## Systematic check
"""

# ╔═╡ 6fbb3423-090b-4e11-912c-902ccb6ab43a
function checkconsistency(σs; ϕ1, θ1, ϕ23)
	(; p1, p2, p3) = constructinvariants(σs)
	p1′ = p1 |> Rz(ϕ23) |> Ry(θ1) |> Rz(ϕ1)
	p2′ = p2 |> Rz(ϕ23) |> Ry(θ1) |> Rz(ϕ1)
	p3′ = p3 |> Rz(ϕ23) |> Ry(θ1) |> Rz(ϕ1)
	(; ϕp, θp, χ) = alteulerangles(p1′,p2′,p3′)
	#
	(ϕ1 ≈ ϕp) * 100 + (θ1 ≈ θp) * 10 + (ϕ23 ≈ χ)
end

# ╔═╡ f66d17ab-3022-4fe4-83de-8604cf8f69fb
σsv = flatDalitzPlotSample(ms; Nev = 10_000)

# ╔═╡ c916fcc4-f570-4358-8030-7ca8de594d8b
prod(checkconsistency.(σsv; ϕ1=-0.3, θ1=π-0.1, ϕ23=-0.4) .== 111)

# ╔═╡ 069f91af-0f1c-45d0-bc29-2f3da4506232
md"""
### Vectors for the crosscheck
"""

# ╔═╡ 3e2a95ce-b7e9-419a-85eb-c025847ae8ae
function rotatedfv(σs; ϕ1, θ1, ϕ23)
	(; p1, p2, p3) = constructinvariants(σs)
	p1′ = p1 |> Rz(ϕ23) |> Ry(θ1) |> Rz(ϕ1)
	p2′ = p2 |> Rz(ϕ23) |> Ry(θ1) |> Rz(ϕ1)
	p3′ = p3 |> Rz(ϕ23) |> Ry(θ1) |> Rz(ϕ1)
	return (p1′,p2′,p3′)
end

# ╔═╡ ec44a76a-0079-4e76-b9f0-b60b649445c9
σs0 = Invariants(ms,
	σ1 = 0.7980703453578917,
	σ2 = 3.6486261122281745)

# ╔═╡ b017b80b-44f0-4cb7-bf9c-c8511fd976cf
with_terminal() do
	println.(rotatedfv(σs0; ϕ1=-0.3, θ1=π-0.1, ϕ23=-0.4))
end

# ╔═╡ 30b38727-cf5b-46f5-8aaf-7b5b7dc0d3a9
p1,p2,p3 = constructinvariants(σs0)

# ╔═╡ 47f54ed9-5f87-4e21-a309-62eddfaeb2e6
begin
	p4p =  [0.        , 0.        , 0.68416827, 1.16122377]
	p4pi =  [-0.28022697,  0.        , -0.1556354 ,  0.34961317]
	p4k =  [ 0.28022697,  0.        , -0.52853287,  0.77562306]
end

# ╔═╡ 0485c5fd-1ee2-4d57-a222-ca31044b1b63
begin
	# Lambda
	pk_theta_r=  1.06382395
	pk_theta_p=  -1.33775751

	# Delta
	ppi_theta_r=  -0.487513
	ppi_theta_p=  1.11390452
	#
	kpi_theta_k=  1.82134117
	#
	# m2ppi:  1.92475412
	# m2pk:  3.64862611
	# m2kpi:  0.79807035
end

# ╔═╡ a588bcfd-fe75-44f7-ac58-88f7039e7584
begin
	kpi_theta_k ≈ acos(cosθ23(σs0,ms²)),
	#
	pk_theta_p ≈ -(π-acos(cosθ31(σs0,ms²))),
	pk_theta_r ≈ π-acos(cosζ(wr(1,2,0),σs0,ms²)),
	#
	ppi_theta_p ≈ acos(cosθ12(σs0,ms²)),
	ppi_theta_r ≈ -(π-acos(cosζ(wr(3,1,0),σs0,ms²)))
end

# ╔═╡ Cell order:
# ╠═b56fc9a7-fa03-47d6-b22e-0097be7155d3
# ╠═d5d6b8bf-1321-464d-80bb-be1cc1b1ed8d
# ╠═d65ed080-9654-11ec-0a8e-e530b9164a8c
# ╠═9beb287e-91ee-4a01-b803-5686576cf087
# ╠═55cf2a64-dd00-461a-ab93-cad2ef71fbb0
# ╠═f341c6fd-4b7a-4497-acf8-f343a95e9521
# ╠═4a5e2b92-37fc-494b-8c00-d8a339ce7bcb
# ╠═570f8ee0-7c75-4c93-bc3a-debab0b81874
# ╠═1a0a3c20-0990-4a7d-8395-941bf34ac9e3
# ╟─34de7c63-83db-4247-b904-1eaa5822af18
# ╠═9a793648-ae8f-465d-9a55-0735a16fd1db
# ╟─89d46224-82a3-4b53-b17e-4fb7ba702202
# ╠═6fbb3423-090b-4e11-912c-902ccb6ab43a
# ╠═f66d17ab-3022-4fe4-83de-8604cf8f69fb
# ╠═c916fcc4-f570-4358-8030-7ca8de594d8b
# ╟─069f91af-0f1c-45d0-bc29-2f3da4506232
# ╠═3e2a95ce-b7e9-419a-85eb-c025847ae8ae
# ╠═ec44a76a-0079-4e76-b9f0-b60b649445c9
# ╠═4f00e2b5-6097-4577-b669-eafe4cbf1852
# ╠═b017b80b-44f0-4cb7-bf9c-c8511fd976cf
# ╠═30b38727-cf5b-46f5-8aaf-7b5b7dc0d3a9
# ╠═47f54ed9-5f87-4e21-a309-62eddfaeb2e6
# ╠═0485c5fd-1ee2-4d57-a222-ca31044b1b63
# ╠═a588bcfd-fe75-44f7-ac58-88f7039e7584

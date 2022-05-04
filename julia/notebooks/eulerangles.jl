### A Pluto.jl notebook ###
# v0.18.0

using Markdown
using InteractiveUtils

# ╔═╡ b56fc9a7-fa03-47d6-b22e-0097be7155d3
begin
	cd(@__DIR__)
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

# ╔═╡ d65ed080-9654-11ec-0a8e-e530b9164a8c
Rz(p,θ) = [p[1]*cos(θ)-p[2]*sin(θ), p[2]*cos(θ)+p[1]*sin(θ), p[3]]

# ╔═╡ 55cf2a64-dd00-461a-ab93-cad2ef71fbb0
Ry(p,θ) = [p[1]*cos(θ)+p[3]*sin(θ), p[2], p[3]*cos(θ)-p[1]*sin(θ)]

# ╔═╡ f341c6fd-4b7a-4497-acf8-f343a95e9521
begin
	Rz(θ) = x->Rz(x,θ)
	Ry(θ) = x->Ry(x,θ)
end

# ╔═╡ 4a5e2b92-37fc-494b-8c00-d8a339ce7bcb
const ms = ThreeBodyMasses(m1=0.938, m2=0.140, m3=0.493, m0=2.28)

# ╔═╡ 570f8ee0-7c75-4c93-bc3a-debab0b81874
const ms² = ms^2

# ╔═╡ f56e88d9-9d91-4275-884c-1c104a3ba016
σ1=1.3

# ╔═╡ 15a58427-99d0-4c4f-80ec-8a83bec659f6
σ2=2.2

# ╔═╡ 973041e0-41ec-4cc8-84c3-ebd500c4ab64
σs = Invariants(ms; σ1, σ2)

# ╔═╡ 1a0a3c20-0990-4a7d-8395-941bf34ac9e3
function constructinvariants(σs)
	E₁ = (ms.m0^2+ms[1]^2-σs[1]) / (2ms.m0)
	p₁ = sqrt(λ(ms.m0^2,ms[1]^2,σs[1])) / (2ms.m0)
	# 
	E₂ = (ms.m0^2+ms[2]^2-σs[2]) / (2ms.m0)
	p₂ = sqrt(λ(ms.m0^2,ms[2]^2,σs[2])) / (2ms.m0)
	# 
	E₃ = (ms.m0^2+ms[3]^2-σs[3]) / (2ms.m0)
	p₃ = sqrt(λ(ms.m0^2,ms[3]^2,σs[3])) / (2ms.m0)
	# 
	p1 = [0,0,p₁]
	p2 = Ry([0,0,p₂], -acos(cosθhat12(σs,ms²)))
	p3 = Ry([0,0,p₃], acos(cosθhat31(σs,ms²)))
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

# ╔═╡ 988777d9-d778-40ab-b1e7-e28cb005793c
md"""
## Example
"""

# ╔═╡ ca76380d-2c28-41b4-86d1-dac42a3385d7
(; p1, p2, p3) = constructinvariants(σs)

# ╔═╡ 381b35a0-4cef-4022-bb2e-97d895e12823
ϕ1, θ1, ϕ23 = 0.3, 1.4, 1.8

# ╔═╡ 08180a31-b8be-4126-8379-b02d0086f238
p1′ = p1 |> Rz(ϕ23) |> Ry(θ1) |> Rz(ϕ1)

# ╔═╡ bd2ab021-8ee3-4f98-8521-03afadb2b31b
p2′ = p2 |> Rz(ϕ23) |> Ry(θ1) |> Rz(ϕ1)

# ╔═╡ a05c4a13-228d-4a93-aba1-dd13070f2da5
p3′ = p3 |> Rz(ϕ23) |> Ry(θ1) |> Rz(ϕ1)

# ╔═╡ c6a307e8-5545-4e09-9028-30ad59c01640
ϕp = atan(p1′[2],p1′[1])

# ╔═╡ c666cb79-6488-407a-a9ee-a1e913ae0188
θp = acos(p1′[3]/norm(p1))

# ╔═╡ 6e9ce703-c346-4d70-a402-4f563bd5914c
χ = atan((p1′ × p3′)[3], -(p1′ × p3′ × p1′)[3] / norm(p1′))

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
σs0 = Invariants(
	σ1 = 0.7980703453578917,
	σ2 = 3.6486261122281745,
	σ3 = 1.894196542413933)

# ╔═╡ b017b80b-44f0-4cb7-bf9c-c8511fd976cf
with_terminal() do
	println.(rotatedfv(σs0; ϕ1=-0.3, θ1=π-0.1, ϕ23=-0.4))
end

# ╔═╡ Cell order:
# ╠═b56fc9a7-fa03-47d6-b22e-0097be7155d3
# ╠═d65ed080-9654-11ec-0a8e-e530b9164a8c
# ╠═55cf2a64-dd00-461a-ab93-cad2ef71fbb0
# ╠═f341c6fd-4b7a-4497-acf8-f343a95e9521
# ╠═4a5e2b92-37fc-494b-8c00-d8a339ce7bcb
# ╠═570f8ee0-7c75-4c93-bc3a-debab0b81874
# ╠═f56e88d9-9d91-4275-884c-1c104a3ba016
# ╠═15a58427-99d0-4c4f-80ec-8a83bec659f6
# ╠═973041e0-41ec-4cc8-84c3-ebd500c4ab64
# ╠═1a0a3c20-0990-4a7d-8395-941bf34ac9e3
# ╟─34de7c63-83db-4247-b904-1eaa5822af18
# ╠═9a793648-ae8f-465d-9a55-0735a16fd1db
# ╟─988777d9-d778-40ab-b1e7-e28cb005793c
# ╠═ca76380d-2c28-41b4-86d1-dac42a3385d7
# ╠═381b35a0-4cef-4022-bb2e-97d895e12823
# ╠═08180a31-b8be-4126-8379-b02d0086f238
# ╠═bd2ab021-8ee3-4f98-8521-03afadb2b31b
# ╠═a05c4a13-228d-4a93-aba1-dd13070f2da5
# ╠═c6a307e8-5545-4e09-9028-30ad59c01640
# ╠═c666cb79-6488-407a-a9ee-a1e913ae0188
# ╠═6e9ce703-c346-4d70-a402-4f563bd5914c
# ╟─89d46224-82a3-4b53-b17e-4fb7ba702202
# ╠═6fbb3423-090b-4e11-912c-902ccb6ab43a
# ╠═f66d17ab-3022-4fe4-83de-8604cf8f69fb
# ╠═c916fcc4-f570-4358-8030-7ca8de594d8b
# ╟─069f91af-0f1c-45d0-bc29-2f3da4506232
# ╠═3e2a95ce-b7e9-419a-85eb-c025847ae8ae
# ╠═ec44a76a-0079-4e76-b9f0-b60b649445c9
# ╠═4f00e2b5-6097-4577-b669-eafe4cbf1852
# ╠═b017b80b-44f0-4cb7-bf9c-c8511fd976cf

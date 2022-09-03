### A Pluto.jl notebook ###
# v0.18.0

using Markdown
using InteractiveUtils

# ╔═╡ 82366e00-987a-11ec-2631-455a8bb3ff96
begin
    cd(joinpath(@__DIR__, ".."))
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

# ╔═╡ fd9ea193-f54b-4ef4-98a5-e00279f48002
const ms = ThreeBodyMasses(0.938, 0.140, 0.493, m0=2.28)

# ╔═╡ fcc14f24-d230-4f79-92bb-85a783ea75c5
begin
    struct minusone end
    import Base: ^
    ^(x::minusone, n::Int) = isodd(n) ? -1 : 1
    macro x_str(s::String)
        minusone()
    end
end

# ╔═╡ 2795ab21-7c3a-44a5-8a34-eb8b04d0c98c
const ms² = ms^2

# ╔═╡ 98515517-7963-45bd-ad45-c824591c283f
const two_jΛ = 1

# ╔═╡ 9c5cab13-e5c2-4b83-876b-6d4f95a8b4f7
function wignerd_angle_doublearg(two_j, two_λ1, two_λ2, θ)
    s, c = sincos(θ)
    wd = wignerd_angle_doublearg(two_j, two_λ1, two_λ2, c)
    return s > 0 ? wd : wd * x"-1"^(two_λ1 - two_λ2)
end

# ╔═╡ e4db3bed-5b1d-49d5-9479-54f12db1332a
function cosαΔ(σs)
    γp_Λc = (ms²[4] + ms²[1] - σs[1]) / (2 * ms[4] * ms[1])
    γΔ_Λc = (ms²[4] + σs[3] - ms²[3]) / (2 * ms[4] * sqrt(σs[3]))
    γp_Δ = (σs[3] + ms²[1] - ms²[2]) / (2 * ms[1] * sqrt(σs[3]))
    (1 + γp_Λc + γΔ_Λc + γp_Δ)^2 / ((1 + γp_Λc) * (1 + γΔ_Λc) * (1 + γp_Δ)) - 1
end

# ╔═╡ 35b9d40e-957d-4c4e-b9f5-169f41097e80
function cosαΛ(σs)
    γp_Λc = (ms²[4] + ms²[1] - σs[1]) / (2 * ms[4] * ms[1])
    γΛ_Λc = (ms²[4] + σs[2] - ms²[2]) / (2 * ms[4] * sqrt(σs[2]))
    γp_Λ = (σs[2] + ms²[1] - ms²[3]) / (2 * ms[1] * sqrt(σs[2]))
    (1 + γp_Λc + γΛ_Λc + γp_Λ)^2 / ((1 + γp_Λc) * (1 + γΛ_Λc) * (1 + γp_Λ)) - 1
end

# ╔═╡ 27b70eff-dd8a-413a-bc1f-6e4cc20d43bf
plot(layout=grid(1, 2), size=(700, 250),
    plot(ms, cosαΔ, colorbar=true),
    plot(ms, cosαΛ, colorbar=true))

# ╔═╡ 8af8de89-9f26-4052-8391-f2733788be5a
function ξΛ(σs)
    θp = -acos(-cosθ31(σs, ms²)) # - (π-θ)
    θΛ = acos(-cosζ12_for0(σs, ms²)) # π-θ
    α = acos(cosαΛ(σs))
    return θp + θΛ + α
end

# ╔═╡ a1c68d06-5d52-4357-b37f-8651a60950da
plot(ms, ξΛ, colorbar=true)

# ╔═╡ e142916e-b305-46c6-8ba3-44c7caeccff8
function ξΔ(σs)
    θp = acos(cosθ12(σs, ms²)) # just θ
    θΔ = -acos(-cosζ31_for0(σs, ms²)) # -(π-θ)
    α = acos(cosαΔ(σs))
    return θp + θΔ - α
end

# ╔═╡ 78a8e0dc-255f-4dc8-8001-90725a3c7b72
plot(layout=grid(1, 2),
    plot(ms, σs -> acos(cosζ13_for1(σs, ms²)) / ξΔ(σs) - 1,
        colorbar=true, title="ξΔ = ζ13_for1"),
    plot(ms, σs -> -acos(cosζ21_for1(σs, ms²)) / ξΛ(σs) - 1,
        colorbar=true, title="ξΛ = - ζ21_for1"))

# ╔═╡ Cell order:
# ╠═82366e00-987a-11ec-2631-455a8bb3ff96
# ╠═fd9ea193-f54b-4ef4-98a5-e00279f48002
# ╠═2795ab21-7c3a-44a5-8a34-eb8b04d0c98c
# ╠═fcc14f24-d230-4f79-92bb-85a783ea75c5
# ╠═98515517-7963-45bd-ad45-c824591c283f
# ╠═9c5cab13-e5c2-4b83-876b-6d4f95a8b4f7
# ╠═e4db3bed-5b1d-49d5-9479-54f12db1332a
# ╠═35b9d40e-957d-4c4e-b9f5-169f41097e80
# ╠═27b70eff-dd8a-413a-bc1f-6e4cc20d43bf
# ╠═8af8de89-9f26-4052-8391-f2733788be5a
# ╠═a1c68d06-5d52-4357-b37f-8651a60950da
# ╠═e142916e-b305-46c6-8ba3-44c7caeccff8
# ╠═78a8e0dc-255f-4dc8-8001-90725a3c7b72

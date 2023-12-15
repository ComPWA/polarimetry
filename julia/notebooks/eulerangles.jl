### A Pluto.jl notebook ###
# v0.19.12

using Markdown
using InteractiveUtils

# ╔═╡ b56fc9a7-fa03-47d6-b22e-0097be7155d3
# ╠═╡ show_logs = false
begin
    cd(joinpath(@__DIR__, ".."))
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
    #
    import YAML
    using Plots
    using RecipesBase
    #
    using LinearAlgebra
    using Parameters
    #
    using ThreeBodyDecay
    using Lc2ppiKSemileptonicModelLHCb
end

# ╔═╡ 049886e5-0839-4dce-9621-32cce58132f5
md"""
# Studies of the polarization angles
"""

# ╔═╡ 814a7e2b-60f5-4775-bed5-8ef54f02252e
theme(:wong2, frame=:box, grid=false, minorticks=true,
    guidefontvalign=:top, guidefonthalign=:right,
    xlim=(:auto, :auto), ylim=(:auto, :auto),
    lw=1, lab="", colorbar=false)

# ╔═╡ 4a5e2b92-37fc-494b-8c00-d8a339ce7bcb
begin
    const ms = Lc2ppiKSemileptonicModelLHCb.ms
    const ms² = ms^2
    ms
end

# ╔═╡ 01777f79-392c-46d2-8dc7-b28cc8a6fc7a
begin
    refs = YAML.load_file(joinpath("..", "data", "observable-references.yaml"))
    pointvariables = Dict{String,Any}(refs["Point cross-check"]["variables"])
end

# ╔═╡ ec44a76a-0079-4e76-b9f0-b60b649445c9
begin
    @unpack m2pk, m2kpi = pointvariables
    σs0 = Invariants(ms, σ1=m2kpi, σ2=m2pk)
end

# ╔═╡ d65ed080-9654-11ec-0a8e-e530b9164a8c
begin
    Rz(p, θ) = [p[1] * cos(θ) - p[2] * sin(θ), p[2] * cos(θ) + p[1] * sin(θ), p[3]]
    Ry(p, cθ, sθ) = [p[1] * cθ + p[3] * sθ, p[2], p[3] * cθ - p[1] * sθ]
    Ry(p, θ) = [p[1] * cos(θ) + p[3] * sin(θ), p[2], p[3] * cos(θ) - p[1] * sin(θ)]

    Rz(θ) = x -> Rz(x, θ)
    Ry(θ) = x -> Ry(x, θ)
    R(ϕ, θ, χ) = x -> R(x, ϕ, θ, χ)

    Ry(ps::NamedTuple{(:p1, :p2, :p3)}, θ) =
        (; p1=Ry(ps.p1, θ), p2=Ry(ps.p2, θ), p3=Ry(ps.p3, θ))
    #
    function R(ps::NamedTuple{(:p1, :p2, :p3)}, ϕ, θ, χ)
        @unpack p1, p2, p3 = ps
        p1′ = p1 |> Rz(χ) |> Ry(θ) |> Rz(ϕ)
        p2′ = p2 |> Rz(χ) |> Ry(θ) |> Rz(ϕ)
        p3′ = p3 |> Rz(χ) |> Ry(θ) |> Rz(ϕ)
        (; p1=p1′, p2=p2′, p3=p3′)
    end
end;

# ╔═╡ 1a0a3c20-0990-4a7d-8395-941bf34ac9e3
"""
	constructaligned(σs)

Constructs four vectors using the aligned configuration-1
"""
function constructinvariants(σs)
    p₁ = sqrtKallenFact(ms.m0, ms[1], sqrt(σs[1])) / (2ms.m0)
    p₂ = sqrtKallenFact(ms.m0, ms[2], sqrt(σs[2])) / (2ms.m0)
    p₃ = sqrtKallenFact(ms.m0, ms[3], sqrt(σs[3])) / (2ms.m0)
    #
    p1 = [0, 0, -p₁]
    p2 = (c = cosζ(wr(1, 2, 0), σs, ms²);
    Ry([0, 0, -p₂], c, -sqrt(1 - c^2)))
    p3 = (c = cosζ(wr(3, 1, 0), σs, ms²);
    Ry([0, 0, -p₃], c, sqrt(1 - c^2)))
    return (; p1, p2, p3)
end

# ╔═╡ 30b38727-cf5b-46f5-8aaf-7b5b7dc0d3a9
begin
    ps = constructinvariants(σs0)
    plot()
    map(enumerate(ps)) do (i, p)
        plot!([0, [1im, 0, 1]' * p], arrow=true, lab="\$p_$i\$", lw=3)
    end
    plot!(xlab="\$p_z\$", ylab="\$p_x\$")
end

# ╔═╡ 34de7c63-83db-4247-b904-1eaa5822af18
md"""
## Check of PAPER-2022-002 relations

The decay amplitude in the LHCb amplitude analysis is written using the DPD,

$T_{m,\lambda}= \sum_\nu D_{m,\nu}^{1/2*}(\phi_p,\theta_p,\chi) O^{\nu}_\lambda(m_{pK}^2, m_{K\pi}^2)$

The overall rotation align the decay vectors with the $xz$ plane, such that $\vec{p}_p \uparrow\uparrow \vec{z}$.
"""

# ╔═╡ 9a793648-ae8f-465d-9a55-0735a16fd1db
function alteulerangles(p1′, p2′, p3′)
    ϕp = atan(p1′[2], p1′[1])
    θp = acos(p1′[3] / norm(p1′))
    χK = atan((p1′×p3′)[3], -(p1′×p3′×p1′)[3] / norm(p1′))
    #
    (; ϕp, θp, χK)
end

# ╔═╡ 8d02d7fd-c45c-4ef1-9deb-c24686eef822
md"""
### Check-1:
Construction of the momenta in the aligned configuration
"""

# ╔═╡ 47f54ed9-5f87-4e21-a309-62eddfaeb2e6
ps0 = constructinvariants(σs0) |> Ry(π)

# ╔═╡ 36526f74-2e4c-44c1-8097-67553277ed83
let
    @unpack p4p, p4pi, p4k = pointvariables
    isapprox(ps0.p1, p4p[1:3]; atol=1e-8),
    isapprox(ps0.p2, p4pi[1:3]; atol=1e-8),
    isapprox(ps0.p3, p4k[1:3]; atol=1e-8)
end |> x -> x => (prod(x) && "✔")

# ╔═╡ b149fad4-efd0-4f03-82fe-0a8ccc587641
md"""
#### Check-2:
Here, we validate that the rotation (ϕp, θp, χ) matches the standard DPD rotation (ϕ1, θ1, ϕ23) up to Ry(π). The plane is the same!
"""

# ╔═╡ 6fbb3423-090b-4e11-912c-902ccb6ab43a
begin
    function checkconsistency(σs; ϕ1, θ1, ϕ23)
        ps = constructinvariants(σs) |> Ry(π)
        @unpack p1, p2, p3 = ps |> R(ϕ1, θ1, ϕ23)
        @unpack ϕp, θp, χK = alteulerangles(p1, p2, p3)
        #
        (ϕ1 ≈ ϕp) * 100 + (θ1 ≈ θp) * 10 + (ϕ23 ≈ χK)
    end
    #
    σsv = flatDalitzPlotSample(ms; Nev=10)
    (checkconsistency.(σsv; ϕ1=-0.3, θ1=1.3, ϕ23=-0.4) .== 111) |>
    x -> x => (prod(x) && "✔")
end

# ╔═╡ c3dada83-3380-4564-aa3c-5bda1632bd80
md"""
#### Check-3:
The angle-mapping relations are checked:
```math
\begin{align}
    &&\bar{\theta}_K &= \theta_{23}\,,\\ \nonumber
    \theta_{\Lambda} &= \pi - \zeta^0_{1(2)}\,,&\theta_p^{\Lambda} &= -(\pi - \theta_{31})\,, \\ \nonumber
    \theta_{\Delta} &= -(\pi-\zeta^0_{3(1)})\,,&\theta_p^{\Delta} &= \theta_{12}\,.
\end{align}
```
"""

# ╔═╡ 30e6011a-a3c6-4080-a097-78424eafd6bc
begin # computed the DPD angles
    θ12 = acos(cosθ12(σs0, ms²))
    θ23 = acos(cosθ23(σs0, ms²))
    θ31 = acos(cosθ31(σs0, ms²))
    #
    ζ⁰12 = acos(cosζ(wr(1, 2, 0), σs0, ms²))
    ζ⁰31 = acos(cosζ(wr(3, 1, 0), σs0, ms²))
end;

# ╔═╡ cca593e6-be26-4d96-bb0c-035e38e9ebb5
begin # get 2022-002 angles
    @unpack pk_theta_r, pk_theta_p = pointvariables
    @unpack ppi_theta_r, ppi_theta_p = pointvariables
    @unpack kpi_theta_k = pointvariables
end;

# ╔═╡ a588bcfd-fe75-44f7-ac58-88f7039e7584
begin # compare
    kpi_theta_k ≈ θ23,
    #
    pk_theta_p ≈ -(π - θ31),
    pk_theta_r ≈ π - ζ⁰12,
    #
    ppi_theta_p ≈ θ12,
    ppi_theta_r ≈ -(π - ζ⁰31)
end |> x -> x => (prod(x) && "✔")

# ╔═╡ 1780255f-65ac-4e82-ba82-0a549b65c4e2
md"""
### Euler angles from the scalar products

Numerical check of the expressions for the angles
```math
\begin{align}
    \phi &= \arctan(-(\vec{p}_P \times \vec{p}_X) \cdot \vec p_D, (-(\vec{p}_P \times \vec{p}_X) \times \frac{-\vec{p}_X}{|\vec{p}_X|}) \cdot \vec p_D)\,,\\ \nonumber
    \theta &= \arccos(\frac{\vec p_X \cdot \vec p_D}{|\vec p_X|\,|\vec p_D|})\,,\\ \nonumber
    \chi &= \arctan((\vec{p}_X \times \vec{p}_D) \cdot \vec p_B, ((\vec{p}_X \times \vec{p}_D) \times \frac{-\vec{p}_D}{|\vec{p}_D|}) \cdot \vec p_B)\,.
\end{align}
```
"""

# ╔═╡ 01618641-7f67-4946-836f-8efe92c7ec36
"""
	eulerfromalgebra(_pD, _pB,
		_pX=[0,0,-1], # -z axes,
		_pP = [1,0,0] # x axes
		)

Computation of the Euler angles for the process A → B+C+D,
produced in the reaction P+Q → A+X
"""
function eulerfromalgebra(_pD, _pB,
    _pX=[0, 0, -1], # -z axes,
    _pP=[1, 0, 0] # x axes
)
    pP, pX, pD, pB = _pP[1:3], _pX[1:3], _pD[1:3], _pB[1:3]
    #
    ϕ = atan(-(pP × pX) · pD, (-(pP × pX) × (-pX) ./ norm(pX)) · pD)
    θ = acos((pX · pD) / norm(pX) / norm(pD))
    χ = atan((pX × pD) · pB, ((pX × pD) × -pD ./ norm(pD)) · pB)
    (; ϕ, θ, χ)
end

# ╔═╡ 6f8deba2-b45d-42e4-9424-ccbd98f9c9c1
let
    angles = (ϕ1=-0.3, θ1=3.0, ϕ23=-0.4)
    pD, pB, pC = constructinvariants(σs0) |> R(angles...)
    ϕθχ = eulerfromalgebra(pD, pB, [0, 0, -1], [1, 0, 0])
    #
    (collect(angles) .≈ collect(ϕθχ))
end |> x -> x => (prod(x) && "✔")

# ╔═╡ eb6ed071-2a2a-41b6-9afa-637e1608ac51
md"""
### CP conjugation
"""

# ╔═╡ a1b0bfdc-84ad-4075-b5f5-c924c0f42dee
md"""
#### First test: flipping $p$, $K$ and $\pi$ momenta in $\Lambda_c^+$ rest frame
The following cell validates the transformation of the angles under parity flip of the three-vectors in the $\Lambda_c^+$ rest frame.
```math
\begin{align}
\phi &\to \left\{
\begin{array}{}
\pi+\phi &\text{ for } \phi < 0,\\
-\pi+\phi &\text{ for } \phi > 0
\end{array}
\right.
\\
\theta&\to \pi-\theta \\
\chi &\to
\left\{
\begin{array}{}
-\pi-\chi &\text{ for } \chi < 0,\\
 \pi-\chi &\text{ for } \chi > 0
\end{array}
\right.
\end{align}
```
"""

# ╔═╡ 060dfd12-7f22-483b-9189-6df247c9850e
begin
    mapϕ(ϕ) = π + ϕ - 2π * (ϕ > 0)
    mapθ(θ) = π - θ
    mapχ(χ) = -π - χ + 2π * (χ > 0)
end;

# ╔═╡ 77eab378-4677-4349-91e3-2d3b81c1e466
let # 3 angles x 10 tries
    mapslices(rand(3, 10), dims=1) do r
        angles = NamedTuple{(:ϕ1, :θ1, :ϕ23)}(r .* [2π, π, 2π] .- [π, 0, π])
        pD, pB, pC = constructinvariants(σs0) |> R(angles...)
        # the xz axes remain, the vectors are flipped
        ϕ, θ, χ = eulerfromalgebra(-pD, -pB, [0, 0, -1], [1, 0, 0])
        #
        mapϕ(angles.ϕ1) ≈ ϕ && mapθ(angles.θ1) ≈ θ && mapχ(angles.ϕ23) ≈ χ
    end
end |> x -> x => (prod(x) && "✔")

# ╔═╡ 7ed18287-8b9f-42b1-8504-1b03bd3c6bda
md"""
#### Second test: flipping all vectors
"""

# ╔═╡ f114f585-bce6-4455-82c8-b9911d861e1a
begin
    mapϕ′(ϕ) = -ϕ
    mapθ′(θ) = θ
    mapχ′(χ) = -χ
end;

# ╔═╡ 23317a51-223e-44f9-b861-5c3d34736d87
let # 3 angles x 10 tries
    mapslices(rand(3, 10), dims=1) do r
        angles = NamedTuple{(:ϕ1, :θ1, :ϕ23)}(r .* [2π, π, 2π] .- [π, 0, π])
        pD, pB, pC = constructinvariants(σs0) |> R(angles...)
        # the xz axes remain, the vectors are flipped
        pD′ = -pD |> Ry(π)
        pB′ = -pB |> Ry(π)
        pX′ = -[0, 0, -1] |> Ry(π)
        pP′ = -[1, 0, 0] |> Ry(π)
        #
        ϕ, θ, χ = eulerfromalgebra(pD′, pB′, pX′, pP′)
        #
        mapϕ′(angles.ϕ1) ≈ ϕ && mapθ′(angles.θ1) ≈ θ && mapχ′(angles.ϕ23) ≈ χ
    end
end |> x -> x => (prod(x) && "✔")

# ╔═╡ Cell order:
# ╟─049886e5-0839-4dce-9621-32cce58132f5
# ╠═b56fc9a7-fa03-47d6-b22e-0097be7155d3
# ╠═814a7e2b-60f5-4775-bed5-8ef54f02252e
# ╠═4a5e2b92-37fc-494b-8c00-d8a339ce7bcb
# ╠═01777f79-392c-46d2-8dc7-b28cc8a6fc7a
# ╠═ec44a76a-0079-4e76-b9f0-b60b649445c9
# ╠═d65ed080-9654-11ec-0a8e-e530b9164a8c
# ╠═1a0a3c20-0990-4a7d-8395-941bf34ac9e3
# ╟─30b38727-cf5b-46f5-8aaf-7b5b7dc0d3a9
# ╟─34de7c63-83db-4247-b904-1eaa5822af18
# ╠═9a793648-ae8f-465d-9a55-0735a16fd1db
# ╟─8d02d7fd-c45c-4ef1-9deb-c24686eef822
# ╠═47f54ed9-5f87-4e21-a309-62eddfaeb2e6
# ╠═36526f74-2e4c-44c1-8097-67553277ed83
# ╟─b149fad4-efd0-4f03-82fe-0a8ccc587641
# ╠═6fbb3423-090b-4e11-912c-902ccb6ab43a
# ╟─c3dada83-3380-4564-aa3c-5bda1632bd80
# ╠═30e6011a-a3c6-4080-a097-78424eafd6bc
# ╠═cca593e6-be26-4d96-bb0c-035e38e9ebb5
# ╠═a588bcfd-fe75-44f7-ac58-88f7039e7584
# ╟─1780255f-65ac-4e82-ba82-0a549b65c4e2
# ╠═01618641-7f67-4946-836f-8efe92c7ec36
# ╠═6f8deba2-b45d-42e4-9424-ccbd98f9c9c1
# ╟─eb6ed071-2a2a-41b6-9afa-637e1608ac51
# ╟─a1b0bfdc-84ad-4075-b5f5-c924c0f42dee
# ╠═060dfd12-7f22-483b-9189-6df247c9850e
# ╠═77eab378-4677-4349-91e3-2d3b81c1e466
# ╟─7ed18287-8b9f-42b1-8504-1b03bd3c6bda
# ╠═f114f585-bce6-4455-82c8-b9911d861e1a
# ╠═23317a51-223e-44f9-b861-5c3d34736d87

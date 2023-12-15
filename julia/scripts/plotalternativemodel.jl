cd(joinpath(@__DIR__, ".."))
using Pkg
Pkg.activate(".")
Pkg.instantiate()
#
import YAML
using Plots
using LaTeXStrings
import Plots.PlotMeasures.mm
#
using Parameters
using Measurements
using DataFrames
using StaticArrays
#
using ThreeBodyDecay
using ThreadsX

using Lc2ppiKSemileptonicModelLHCb


theme(:wong2, frame=:box, grid=false, minorticks=true,
    guidefontvalign=:top, guidefonthalign=:right,
    xlim=(:auto, :auto), ylim=(:auto, :auto),
    lw=1, lab="", colorbar=false)



#                                  _|            _|
#  _|_|_|  _|_|      _|_|      _|_|_|    _|_|    _|
#  _|    _|    _|  _|    _|  _|    _|  _|_|_|_|  _|
#  _|    _|    _|  _|    _|  _|    _|  _|        _|
#  _|    _|    _|    _|_|      _|_|_|    _|_|_|  _|


const model = published_model("Default amplitude model")
const model16 = published_model("Alternative amplitude model with K(1430) with free width");
#

let
    k = "K(1430)"
    #
    i = findfirst(x -> x == k, model.isobarnames)
    i′ = findfirst(x -> x == k, model16.isobarnames)
    #
    BW = model.chains[i].Xlineshape
    BW′ = model16.chains[i′].Xlineshape
    #
    @unpack m1, m2, m0, mk = BW
    xv = range((m1 + m2)^2, (m0 - mk)^2, length=300)
    #
    lsh(BW, σ) = abs2(BW(σ)) *
                 Lc2ppiKSemileptonicModelLHCb.breakup(σ, m1^2, m2^2) *
                 Lc2ppiKSemileptonicModelLHCb.breakup(m0^2, σ, mk^2) / sqrt(σ)
    #
    yv = map(xv) do x
        lsh(BW, x)
    end
    yv′ = map(xv) do x
        lsh(BW′, x)
    end
    #
    plot(xv, yv, lab=k)
    plot!(xv, yv′, lab="I′/I = $(round(sum(yv′)/sum(yv); digits=2))")
end


# projections

pdata = flatDalitzPlotSample(ms; Nev=100_000);

@time const Aiv = ThreadsX.collect(
    SVector([amplitude(d, σs, two_λs) for d in model.chains])
    for two_λs in itr(tbs.two_js), σs in pdata);
#
Iv = sum(Aiv; dims=(1, 2, 3, 4)) do x
    abs2(sum(x .* model.couplings))
end[1, 1, 1, 1, :]

#
@time const Aiv′ = ThreadsX.collect(
    SVector([amplitude(d, σs, two_λs) for d in model16.chains])
    for two_λs in itr(tbs.two_js), σs in pdata);
#
Iv′ = sum(Aiv′; dims=(1, 2, 3, 4)) do x
    abs2(sum(x .* model16.couplings))
end[1, 1, 1, 1, :]


let
    bins = 150
    plot(layout=grid(1, 3), size=(800, 300), yaxis=nothing,
        ylab=L"\mathrm{rate}\,\,(\mathrm{a.u.})",
        left_margin=5mm, bottom_margin=7mm)
    stephist!(sp=1, getproperty.(pdata, :σ2), weights=Iv; bins,
        xlab=L"m^2(pK^-)\,(\mathrm{GeV}^2)")
    stephist!(sp=2, getproperty.(pdata, :σ1), weights=Iv; bins,
        xlab=L"m^2(K^-\pi^+)\,(\mathrm{GeV}^2)", lab="detault model")
    stephist!(sp=3, getproperty.(pdata, :σ3), weights=Iv; bins,
        xlab=L"m^2(p\pi^-)\,(\mathrm{GeV}^2)")
    # model 16
    stephist!(sp=1, getproperty.(pdata, :σ2), weights=Iv′; bins,
        xlab=L"m^2(pK^-)\,(\mathrm{GeV}^2)")
    stephist!(sp=2, getproperty.(pdata, :σ1), weights=Iv′; bins,
        xlab=L"m^2(K^-\pi^+)\,(\mathrm{GeV}^2)", lab="model-16")
    stephist!(sp=3, getproperty.(pdata, :σ3), weights=Iv′; bins,
        xlab=L"m^2(p\pi^-)\,(\mathrm{GeV}^2)")
end

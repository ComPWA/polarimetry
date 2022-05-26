cd(joinpath(@__DIR__, ".."))
using Pkg
Pkg.activate(".")
Pkg.instantiate()
#
using JSON
using Plots
using LaTeXStrings
import Plots.PlotMeasures.mm
#
using Parameters
using Measurements
using DataFrames
#
using ThreeBodyDecay

using Lb2ppiKModelLHCb


theme(:wong2, frame=:box, grid=false, minorticks=true,
    guidefontvalign=:top, guidefonthalign=:right,
    xlim=(:auto, :auto), ylim=(:auto, :auto),
    lw=1, lab="", colorbar=false)



#                                  _|            _|
#  _|_|_|  _|_|      _|_|      _|_|_|    _|_|    _|
#  _|    _|    _|  _|    _|  _|    _|  _|_|_|_|  _|
#  _|    _|    _|  _|    _|  _|    _|  _|        _|
#  _|    _|    _|    _|_|      _|_|_|    _|_|_|  _|

isobarsinput = readjson(joinpath("..", "data", "isobars.json"))["isobars"];
#
isobars = Dict()
for (key, dict) in isobarsinput
    isobars[key] = buildchain(key, dict)
end





modelparameters =
    readjson(joinpath("..", "data", "modelparameters.json"))["modelstudies"];

defaultparameters = first(modelparameters)["parameters"]
shapeparameters = filter(x -> x[1] != 'A', keys(defaultparameters))

parameterupdates = [ # 6 values are updated
    "K(1430)" => (γ=eval(Meta.parse(defaultparameters["gammaK(1430)"])).val,),
    "K(700)" => (γ=eval(Meta.parse(defaultparameters["gammaK(700)"])).val,),
    "L(1520)" => (m=eval(Meta.parse(defaultparameters["ML(1520)"])).val,
        Γ=eval(Meta.parse(defaultparameters["GL(1520)"])).val),
    "L(2000)" => (m=eval(Meta.parse(defaultparameters["ML(2000)"])).val,
        Γ=eval(Meta.parse(defaultparameters["GL(2000)"])).val)]
#
@assert length(shapeparameters) == 6

# plot shapes vs updates
let
    plot(layout=grid(1, length(parameterupdates)), size=(700, 200))
    for (sp, (p, u)) in enumerate(parameterupdates)
        BW = isobars[p].Xlineshape
        plot!(BW; sp, lab=p)
        plot!(updatepars(BW, merge(BW.pars, u)); sp)
    end
    plot!()
end
savefig(joinpath("plots", "updatedefaultmodel.pdf"))


# apply updates
for (p, u) in parameterupdates
    BW = isobars[p].Xlineshape
    isobars[p] = merge(isobars[p], (Xlineshape=updatepars(BW, merge(BW.pars, u)),))
end

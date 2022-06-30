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
#
using ThreeBodyDecay

using Lc2ppiKModelLHCb


theme(:wong2, frame=:box, grid=false, minorticks=true,
    guidefontvalign=:top, guidefonthalign=:right,
    xlim=(:auto, :auto), ylim=(:auto, :auto),
    lw=1, lab="", colorbar=false)



#                                  _|            _|
#  _|_|_|  _|_|      _|_|      _|_|_|    _|_|    _|
#  _|    _|    _|  _|    _|  _|    _|  _|_|_|_|  _|
#  _|    _|    _|  _|    _|  _|    _|  _|        _|
#  _|    _|    _|    _|_|      _|_|_|    _|_|_|  _|



# 1) get isobars
isobarsinput = YAML.load_file(joinpath("..", "data", "particle-definitions.yaml"))
modelparameters =
    YAML.load_file(joinpath("..", "data", "model-definitions.yaml"))
defaultmodel = modelparameters["Default amplitude model"]
isobars = Dict()
for (key, lineshape) in defaultmodel["lineshapes"]
    dict = Dict{String,Any}(isobarsinput[key])
    dict["lineshape"] = lineshape
    isobars[key] = buildchain(key, dict)
end

# 2) update model parameters
defaultparameters = defaultmodel["parameters"]
defaultparameters["ArK(892)1"] = "1.0 ± 0.0"
defaultparameters["AiK(892)1"] = "0.0 ± 0.0"
#
shapeparameters = filter(x -> x[1] != 'A', keys(defaultparameters))
#
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

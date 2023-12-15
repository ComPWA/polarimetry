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



(; particledict, modelparameters) = expose_model_description()
defaultparameters = modelparameters["Default amplitude model"]

const model = Lc2ppiKModel(;
    parse_model_dictionaries(defaultparameters; particledict)...)

# unupdated
shapeparameters = filter(x -> x[1] != 'A', keys(defaultparameters["parameters"]))
for s in shapeparameters
    pop!(defaultparameters["parameters"], s)
end

const model_unupdated = Lc2ppiKModel(;
    parse_model_dictionaries(defaultparameters; particledict)...)

# plot shapes vs updates
whichisobarsmodified =
    Set(getproperty.(parseshapedparameter.(shapeparameters), :isobarname))
let
    plot(layout=grid(1, length(whichisobarsmodified)), size=(700, 200))
    for (sp, k) in enumerate(whichisobarsmodified)
        i = findfirst(x -> x == k, model.isobarnames)
        i′ = findfirst(x -> x == k, model_unupdated.isobarnames)
        BW = model.chains[i].Xlineshape
        BW′ = model_unupdated.chains[i′].Xlineshape
        plot!(BW′; sp, lab=k)
        plot!(BW; sp)
    end
    plot!()
end

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

using BenchmarkTools
using InteractiveUtils

#                                  _|            _|
#  _|_|_|  _|_|      _|_|      _|_|_|    _|_|    _|
#  _|    _|    _|  _|    _|  _|    _|  _|_|_|_|  _|
#  _|    _|    _|  _|    _|  _|    _|  _|        _|
#  _|    _|    _|    _|_|      _|_|_|    _|_|_|  _|

const model0 = let
    # 1) get isobars
    isobarsinput = readjson(joinpath("..", "data", "isobars.json"))["isobars"]
    #
    isobars = Dict()
    for (key, dict) in isobarsinput
        isobars[key] = buildchain(key, dict)
    end

    # 2) update model parameters
    modelparameters =
        readjson(joinpath("..", "data", "modelparameters.json"))["modelstudies"]

    defaultparameters = first(modelparameters)["parameters"]
    defaultparameters["ArK(892)1"] = "1.0 ± 0.0"
    defaultparameters["AiK(892)1"] = "0.0 ± 0.0"

    parameterupdates = [ # 6 values are updated
        "K(1430)" => (γ=eval(Meta.parse(defaultparameters["gammaK(1430)"])).val,),
        "K(700)" => (γ=eval(Meta.parse(defaultparameters["gammaK(700)"])).val,),
        "L(1520)" => (m=eval(Meta.parse(defaultparameters["ML(1520)"])).val,
            Γ=eval(Meta.parse(defaultparameters["GL(1520)"])).val),
        "L(2000)" => (m=eval(Meta.parse(defaultparameters["ML(2000)"])).val,
            Γ=eval(Meta.parse(defaultparameters["GL(2000)"])).val)]

    # apply updates
    for (p, u) in parameterupdates
        BW = isobars[p].Xlineshape
        isobars[p] = merge(isobars[p], (Xlineshape=updatepars(BW, merge(BW.pars, u)),))
    end

    # 3) get couplings
    couplingkeys = collect(filter(x -> x[1:2] == "Ar", keys(defaultparameters)))

    terms = []
    for parname in couplingkeys
        c_re_key = "Ar" * parname[3:end] # = parname
        c_im_key = "Ai" * parname[3:end]
        value_re = eval(Meta.parse(defaultparameters[c_re_key])).val
        value_im = eval(Meta.parse(defaultparameters[c_im_key])).val
        value = value_re + 1im * value_im
        #
        c0, d = parname2decaychain(parname, isobars)
        #
        push!(terms, (c0 * value, d))
    end

    chains = getindex.(terms, 2)
    couplings = getindex.(terms, 1)

    (; chains, couplings)
end

const λσs0 = randomPoint(tbs)

# check that the type is correctly inferred
@code_warntype amplitude(λσs0, model0.chains[3])

# run several times, get time
@btime amplitude.($(Ref(λσs0)), $(model0.chains))

# full information
@benchmark amplitude.($(Ref(λσs0)), $(model0.chains))

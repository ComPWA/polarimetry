cd(joinpath(@__DIR__, ".."))
using Pkg
Pkg.activate(".")
Pkg.instantiate()
#
import YAML
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

using Lc2ppiKSemileptonicModelLHCb

using BenchmarkTools
using InteractiveUtils

#                                  _|            _|
#  _|_|_|  _|_|      _|_|      _|_|_|    _|_|    _|
#  _|    _|    _|  _|    _|  _|    _|  _|_|_|_|  _|
#  _|    _|    _|  _|    _|  _|    _|  _|        _|
#  _|    _|    _|    _|_|      _|_|_|    _|_|_|  _|

const model0 = published_model("Default amplitude model")

const λσs0 = randomPoint(tbs)

# check that the type is correctly inferred
@code_warntype amplitude(λσs0, model0.chains[3])

# run several times, get time
@btime amplitude.($(model0.chains), $(Ref(λσs0)))

# full information
@benchmark amplitude.($(model0.chains), $(Ref(λσs0)))

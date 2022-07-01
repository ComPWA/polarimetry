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

using Lc2ppiKModelLHCb

using BenchmarkTools
using InteractiveUtils

#                                  _|            _|
#  _|_|_|  _|_|      _|_|      _|_|_|    _|_|    _|
#  _|    _|    _|  _|    _|  _|    _|  _|_|_|_|  _|
#  _|    _|    _|  _|    _|  _|    _|  _|        _|
#  _|    _|    _|    _|_|      _|_|_|    _|_|_|  _|

isobarsinput = YAML.load_file(joinpath("..", "data", "particle-definitions.yaml"));

modelparameters =
    YAML.load_file(joinpath("..", "data", "model-definitions.yaml"));

const model0 = LHCbModel(
    modelparameters["Default amplitude model"];
    particledict=isobarsinput)

const λσs0 = randomPoint(tbs)

# check that the type is correctly inferred
@code_warntype amplitude(λσs0, model0.chains[3])

# run several times, get time
@btime amplitude.($(Ref(λσs0)), $(model0.chains))

# full information
@benchmark amplitude.($(Ref(λσs0)), $(model0.chains))

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
    isobars[key] = definechaininputs(key, dict)
end

# 2) update model parameters
defaultparameters = defaultmodel["parameters"]
defaultparameters["ArK(892)1"] = "1.0 ± 0.0"
defaultparameters["AiK(892)1"] = "0.0 ± 0.0"
#
shapeparameters = filter(x -> x[1] != 'A', keys(defaultparameters))
#
parameterupdates = [
    "K(1430)" => (γ=MeasuredParameter(defaultparameters["gammaK(1430)"]).val,),
    "K(700)" => (γ=MeasuredParameter(defaultparameters["gammaK(700)"]).val,),
    "L(1520)" => (m=MeasuredParameter(defaultparameters["ML(1520)"]).val,
        Γ=MeasuredParameter(defaultparameters["GL(1520)"]).val),
    "L(2000)" => (m=MeasuredParameter(defaultparameters["ML(2000)"]).val,
        Γ=MeasuredParameter(defaultparameters["GL(2000)"]).val)]
#
@assert length(shapeparameters) == 6

# apply updates
for (p, u) in parameterupdates
    BW = isobars[p].Xlineshape
    isobars[p] = merge(isobars[p], (Xlineshape=updatepars(BW, merge(BW.pars, u)),))
end


#        _|              _|
#    _|_|_|    _|_|_|  _|_|_|_|    _|_|_|
#  _|    _|  _|    _|    _|      _|    _|
#  _|    _|  _|    _|    _|      _|    _|
#    _|_|_|    _|_|_|      _|_|    _|_|_|


crosscheckresult = readjson(joinpath("..", "data", "crosscheck.json"));

σs0 = Invariants(ms,
    σ1=crosscheckresult["chainvars"]["m2kpi"],
    σ2=crosscheckresult["chainvars"]["m2pk"])

parsepythoncomplex(s::String) = eval(Meta.parse(
    replace(s,
        "(" => "",
        ")" => "",
        "j" => "im")))

begin
    tfK892BW0 = crosscheckresult["lineshapes"]["BW_K(892)_p^1_q^0"] |>
                parsepythoncomplex
    myK892BW0 = parname2decaychain("ArK(892)1", isobars)[2].Xlineshape(σs0[1])
    @assert myK892BW0 ≈ tfK892BW0
end

begin
    myL1405BW0 = parname2decaychain("ArL(1405)1", isobars)[2].Xlineshape(σs0[2])
    tfL1405BW0 = crosscheckresult["lineshapes"]["BW_L(1405)_p^0_q^0"] |>
                 parsepythoncomplex
    @assert tfL1405BW0 ≈ myL1405BW0
end

begin
    myL1690BW0 = parname2decaychain("ArL(1690)1", isobars)[2].Xlineshape(σs0[2])
    tfL1690BW0 = crosscheckresult["lineshapes"]["BW_L(1690)_p^2_q^1"] |>
                 parsepythoncomplex
    @assert tfL1690BW0 ≈ myL1690BW0
end


Adict2matrix(d::Dict) = parsepythoncomplex.(
    [d["A++"] d["A+-"]
        d["A-+"] d["A--"]])
#

crosscheckresult_realpars = filter(kv -> kv[1][2] == 'r', crosscheckresult["chains"])

comparison = DataFrame()
for (parname, adict) in crosscheckresult_realpars
    c, d = parname2decaychain(parname, isobars)
    M_DPD = [c * amplitude(d, σs0, [two_λ1, 0, 0, two_λ0])
             for (two_λ0, two_λ1) in [(1, 1) (1, -1)
        (-1, 1) (-1, -1)]]
    M_LHCb′ = amplitudeLHCb2DPD(Adict2matrix(adict))
    #
    r = filter(x -> !(isnan(x)), vcat(M_DPD ./ M_LHCb′))
    push!(comparison, (; parname=parname[3:end], r, M_DPD, M_LHCb′))
end
sort!(comparison, order(:parname, by=x -> x[3:6]))
sort!(comparison,
    order(:parname, by=x -> findfirst(x[1], "LDK")))


@assert maximum(abs.(vcat(comparison.r...) .- 1)) < 1e-4


select(
    transform(
        transform(comparison,
            :parname => ByRow(x -> isobars[x[1:end-1]].Xlineshape.l) => :l),
        :parname => ByRow(x -> isobars[x[1:end-1]].Xlineshape.minL) => :Lmin),
    [:parname, :r, :l, :Lmin])

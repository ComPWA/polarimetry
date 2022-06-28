module Lc2ppiKModelLHCb

using JSON
#
using LinearAlgebra
using Parameters
using Measurements
using DataFrames
#
using ThreeBodyDecay
#
using RecipesBase



export ms, tbs, parities
include("kinematics.jl")


export Lineshape
export BreitWignerMinL
export BuggBreitWignerMinL
export Flatte1405
export updatepars
include("lineshapes.jl")

export selectindexmap
export couplingLHCb2DPD
export amplitudeLHCb2DPD
export parname2decaychain
include("mapping.jl")

export buildchain
export readjson
include("io.jl")


export intensity
include("amplitude.jl")


export two_Δλ
export σPauli
export twoλ2ind
export expectation
include("sensitivity.jl")


end # module

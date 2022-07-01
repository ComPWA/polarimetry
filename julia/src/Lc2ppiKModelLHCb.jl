module Lc2ppiKModelLHCb

using JSON
#
using LinearAlgebra
using StaticArrays
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

export definechaininputs
export readjson
export parseshapedparameter
export replacementpair
include("io.jl")


export intensity
export LHCbModel
include("amplitude.jl")


export two_Δλ
export σPauli
export twoλ2ind
export expectation
include("sensitivity.jl")


end # module

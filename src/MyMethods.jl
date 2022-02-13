module MyMethods

# Dependencies =================================================================
using LinearAlgebra
using Random
using Distributions
using DataFrames
using JuMP
using GLPK

# Module export ================================================================

export myEstimator, myLS, myTSLS, mySieve, myQR
export coef, predict, inference # my methods
export get_basis # helper functions

# Module content ===============================================================

include("FUN/myEstimator.jl") # myEstimators

include("FUN/myLS.jl") # myLS

include("FUN/myTSLS.jl") # myLS

include("FUN/mySieve.jl") # mySieve

include("FUN/myQR.jl") # myQR

end

module MyMethods

# Dependencies =================================================================
using LinearAlgebra
using Random
using Distributions
using DataFrames

# Module export ================================================================

export myEstimator, myLS, myTSLS
export coef, predict, inference # my methods

# Module content ===============================================================

include("FUN/myEstimator.jl") # myEstimators

include("FUN/myLS.jl") # myLS

include("FUN/myTSLS.jl") # myLS

end

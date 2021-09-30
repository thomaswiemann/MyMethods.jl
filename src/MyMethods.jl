module MyMethods

# Dependencies =================================================================
using LinearAlgebra
using Random
using Distributions
using DataFrames

# Module export ================================================================

export myLS
export coef, predict, inference # my methods

# Module content ===============================================================

include("FUN/myLS.jl") # myLS

end

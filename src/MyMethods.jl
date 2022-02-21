module MyMethods

# Dependencies =================================================================
using LinearAlgebra
using Random
using Distributions
using DataFrames
using JuMP
using GLPK

# Module export ================================================================

export myEstimator, myLS, myTSLS, mySieve, myQR, myLLR, myLQR, myPQR, myMTA
export coef, predict, inference # my methods
export get_basis, get_kw, get_Silverman # helper functions

# Module content ===============================================================

include("FUN/myEstimator.jl") # myEstimators

include("FUN/myLS.jl") # myLS

include("FUN/myTSLS.jl") # myTSLS

include("FUN/mySieve.jl") # mySieve

include("FUN/myQR.jl") # myQR

include("FUN/myLLR.jl") # myLLR

include("FUN/myLQR.jl") # myLQR

include("FUN/myPQR.jl") # myPQR

include("FUN/myMTA.jl") # myMTA

end

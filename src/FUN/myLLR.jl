"""
myLLR(y, x, x_0, K, h; kernel = "Epanechnikov", control = nothing)

A local linear regression implementation.
"""
struct myLLR <: myEstimator
    β::Array{Float64,2} # coefficients
    y::Array{Float64,2} # response
    X # matrix of regressors
    x # running variable
    x0 # value of x at which to evaluate the kernel
    K::Int64 # degree of local linear regression
	h::Float64 # bandwidth
	kernel # kernel function
	control # additional variables included in LLR
    
    # Define constructor function
    function myLLR(y, x, x0, K, h; kernel = "Epanechnikov", control = nothing)	

        # Calculate kernel weights
        u = (x0 .- x) ./ h
        w = get_kw(u, kernel)
        w = w ./ sum(w) # normalize

        # Create regressor matrix
        X = reduce(hcat, [u.^d for d in 0:K])
        if !isnothing(control)
            X = hcat(X, control)
        end

        # Calculate weighted least squares fit
        β = myLS(y, X, weights = w).β

        # Organize and return output
        new(β, y, X, x, x0, K, h, kernel, control)
    end #MYLLR
end #MYLLR

# Helper Functions =============================================================
"""
get_kw(u, kernel)

Function to construct kernel weights. 
"""
function get_kw(u, kernel)
    # Calculate kernel weights
    w = Array{Float64,2}(undef, length(u), 1)
    if kernel == "Epanechnikov"
        @. w = (1 - u^2) * (abs(u) < 1)
    elseif kernel == "Uniform"
        @. w = (abs(u) < 1)
    elseif kernel == "Biweight"
        @. w = (1-u^2)^2
    elseif kernel == "Gaussian"
        w = pdf(Normal(), u.^2)
    else
        println("Specified kernel not (yet) implemented.")
    end
    # Rreturn weights
    return w 
end #GET_KW

"""
get_Silverman(u, kernel)

Function to obtain rule-of-thumb for bandwidth based on Silverman (1986).
"""
function get_Silverman(x; kernel = "Epanechnikov")
    # Data parameters
    nobs = length(x)
    σ = std(x)
    # Get \delta
    which_kernel = findfirst(kernel .== ["Uniform", "Epanechnikov", "Triangular", 
            "Biweight", "Gaussian", "Mahalanobis"])
    δ = [1.351, 1.7188, NaN, 2.0362, 0.7764, 1][which_kernel]
    # Calculate and return rule-of-thumb bandwidth
    h = 1.3643 * δ * σ * (nobs)^(-0.2)
    return h
end #MYSILVERMAN
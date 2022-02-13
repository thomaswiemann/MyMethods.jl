"""
myQR(y, X, τ = 0.5; weights = nothing)

A simple implementation of quantile regression. The argument ``weights`` allows
    specification of weighted loss functions. Must be a vector equal to the 
    number of observations.
"""
struct myQR <: myEstimator
    β::Array{Float64} # coefficients
    y::Array{Float64} # responses
    X::Array{Float64} # matrix of regressors
    τ # quantile
    lp_model # a JuMP model 
    weights # (Optional) vector of weights for loss function

    function myQR(y, X, τ = 0.5; weights = nothing)
        
        # Data parameters
        nobs = length(y)
        nX = size(X, 2)

        # Initialize the linear program
        lp_model = Model(GLPK.Optimizer)

        # Initialize decision variables
        @variable(lp_model, uv[1:nobs, 1:2] >= 0) # residuals
        @variable(lp_model, β_τ[1:nX])

        # Formulate constraint
        @constraint(lp_model, c[i = 1:nobs], 
            X[i, :]' * β_τ + uv[i, 1] - uv[i, 2] == y[i])

        # Define the check-loss function
        if isnothing(weights)
            weights = 1
        end
        @objective(lp_model, Min, 
            sum(weights .* (τ * uv[1:nobs, 1] + (1 - τ) * uv[1:nobs, 2])))

        # Surpress output and solve the linear program
        MOI.set(lp_model, MOI.Silent(), true)
        optimize!(lp_model)

        # Set β
        β = value.(lp_model[:β_τ])

        # Define output
        new(β, y, X, τ, lp_model, weights)
    end #MYQR
end #MYQR

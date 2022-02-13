"""
myPQR(y, X, τ = 0.5, λ = 1; weights = nothing)

A simple implementation of penalized quantile regression. 
"""
struct myPQR 
    α::Array{Float64} # coefficients
    y::Array{Float64} # responses
    x # regressor
    τ # quantile
    λ # penalty parameter
    lp_model # a JuMP model 
    weights # (Optional) vector of weights for loss function

    function myPQR(y, x, λ, τ = 0.5; weights = nothing)
        
        # Data parameters
        nobs = length(y)
        obs_order = sortperm(x[:])

        # Order data
        y_o = y[obs_order]
        x_o = x[obs_order]
        h = x_o[2:nobs] - x_o[1:(nobs - 1)] # lags

        # Initialize the linear program
        lp_model = Model(GLPK.Optimizer)

        # Initialize decision variables
        @variable(lp_model, uv[1:nobs, 1:2] >= 0) # residuals
        @variable(lp_model, α[1:nobs])
        @variable(lp_model, Δ[1:(nobs - 1)]) # differences in beta
        @variable(lp_model, V[1:(nobs - 1)]) # smoothing loss

        # Formulate constraints
        @constraint(lp_model, c[i = 1:nobs], 
            α[i] + uv[i, 1] - uv[i, 2] == y_o[i])
        @constraint(lp_model, penalty[i = 1:(nobs - 2)], 
            (α[i + 2] - α[i + 1]) / h[i + 1] - (α[i + 1] - α[i]) / h[i] == Δ[i])
        @constraint(lp_model, abs_penalty[i = 1:(nobs - 1), j = 0:1], 
            V[i] >= (-1)^j * Δ[i])

        # Define the check-loss function
        if isnothing(weights)
            weights = 1
        end
        @objective(lp_model, Min, 
            sum(weights .* (τ * uv[1:nobs, 1] + (1 - τ) * uv[1:nobs, 2])) + 
            λ * sum(V))

        # Surpress output and solve the linear program
        MOI.set(lp_model, MOI.Silent(), true)
        optimize!(lp_model)

        # Set α
        α = value.(lp_model[:α])

        # Define output
        new(α, y, x, τ, λ, lp_model, weights)
    end #MYPQR
end #MYPQR

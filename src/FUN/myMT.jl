"""
myMTA(y, X, τ = 0.5; weights = nothing)

An implementation of the Mass Transport Approach proposed in  Chiong, Galichon, 
    & Shum (2016). 
"""
struct myMTA <: myEstimator
    β::Array{Float64} # coefficients
    y::Array{Float64} # responses
    X::Array{Float64} # matrix of regressors
    τ # quantile
    lp_model # a JuMP model 
    weights # (Optional) vector of weights for loss function

    function myMTA(y, X; τ = 0.5, weights = nothing)
        

        ccp 

        # Data parameters
        J = length(ccp)

        S = 1000
        μ = -0.57721
        σ = 1
        ϵ = rand(GeneralizedExtremeValue(μ, σ, 0), (S, J))

        # Initialize the linear program
        lp_model = Model(GLPK.Optimizer)

        # Initialize decision variables
        @variable(lp_model, λ[1:J]) 
        @variable(lp_model, z[1:S])

        # Formulate constraint
        @constraint(lp_model, c[s = 1:S, j = 1:J], 
            λ[j] + z[s] >= ϵ[s, j])

        # Define the objective function
        @objective(lp_model, Min, sum(ccp .* λ) + mean(z))

        # Surpress output and solve the linear program
        MOI.set(lp_model, MOI.Silent(), true)
        optimize!(lp_model)

        # Set β
        w = value.(lp_model[:λ]) .+ mean(value.(lp_model[:z]))
        # Define output
        new(β, y, X, τ, lp_model, weights)
    end #MYMTA
end #MYMTA
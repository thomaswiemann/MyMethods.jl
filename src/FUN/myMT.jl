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

    function myMTA(CCP, X, P; S = 100τ = 0.5, weights = nothing)
        
        

        # Data parameters
        T, J = size(CCP)
        nX = size(X, 3)

        S = 100
        
        ϵ = rand(P, (T, J + 1, S))

        # Initialize the linear program
        lp_model = Model(GLPK.Optimizer)

        # Initialize decision variables
        @variable(lp_model, λ[1:T, 1:(J + 1)]) 
        @variable(lp_model, z[1:T, 1:S])
        @variable(lp_model, β[1:nX])
        @variable(lp_model, α[1:J])

        # Formulate constraint
        @constraint(lp_model, c1[t = 1:T, j = 1:J],
            λ[t, j] == α[j] + X[t, j, :]' * β)
        @constraint(lp_model, c2[t = 1:T],
            λ[t, J + 1] == 0)
        @constraint(lp_model, c[t = 1:T, j = 1:(J +1), s = 1:S], 
            λ[t, j] + z[t, s] >= ϵ[t, j, s])

        # Define the objective function
        @objective(lp_model, Min, (sum(CCP .* λ[:, 1:J]) + sum(z) / S) / T)

        # Surpress output and solve the linear program
        MOI.set(lp_model, MOI.Silent(), true)
        optimize!(lp_model)

        # Set slope β and intercepts α
        β = -value.(lp_model[:β])
        α = -value.(lp_model[:α])
        # Define output
        new(β, y, X, τ, lp_model, weights)
    end #MYMTA
end #MYMTA
"""
myMTA(ccp, Q; S = 100)

An implementation of the Mass Transport Approach proposed in  Chiong, Galichon, 
    & Shum (2016) based on Proposition 2. 
"""
struct myMTA
    w::Array{Float64} # slope coefficients
    ccp::Array{Float64} # conditional choice probabilities
    Q # latent utility shock distribution
    S # number of samples from Q
    ∂Gstar # value of the subdifferential at the ccps
    ϵ # Array of epsilon draws from Q
    lp_model # a JuMP model 

    function myMTA(ccp; Q = nothing, S = 100, ϵ = nothing)
        # Data parameters
        J = length(ccp)

        # Sample iid from Q
        if isnothing(ϵ)
            ϵ = rand(Q, (J, S))
        else
            S = size(ϵ, 2)
        end

        # Initialize the linear program
        lp_model = Model(GLPK.Optimizer)

        # Initialize decision variables
        @variable(lp_model, λ[1:J]) 
        @variable(lp_model, z[1:S])

        # Formulate constraint
        @constraint(lp_model, c[j = 1:J, s = 1:S], 
            λ[j] + z[s] >= ϵ[j, s])
        @constraint(lp_model, w0, 
            λ[1] == 0)

        # Define the objective function
        @objective(lp_model, Min, sum(ccp .* λ) + sum(z) / S )

        # Surpress output and solve the linear program
        MOI.set(lp_model, MOI.Silent(), true)
        optimize!(lp_model)

        # Set systematic utilities
        w = -value.(lp_model[:λ])

        # Set objective value
        ∂Gstar = objective_value(lp_model)

        # Define output
        new(w, ccp, Q, S, ∂Gstar, ϵ, lp_model)
    end #MYMTA
end #MYMTA

# Helper Functions =============================================================
"""
get_mtabounds(ccp, ϵ, ∂Gstar; S = 100)

An implementation of the Mass Transport Approach proposed in  Chiong, Galichon, 
    & Shum (2016) based on Theorem 4. 
"""
function get_mtabounds(fit::myMTA)
    # Data parameters
    J, S = size(fit.ϵ)
    
    # Define matrix with bounds on w_y, ∀y
    bounds = zeros((J, 2))

    # Run nested loop to calculate upper and lower bounds
    for j = 2:J
        # Select the product
        f_y = zeros(J)
        f_y[j] = 1
        # Calculate lower/upper bound
        for p in 1:2
            sign_y = (-1)^(p - 1)

            # Initialize the linear program
            lp_model = Model(GLPK.Optimizer)

            # Initialize decision variables
            @variable(lp_model, λ[1:J]) 
            @variable(lp_model, z[1:S])

            # Formulate constraint
            @constraint(lp_model, c[j = 1:J, s = 1:S], 
                λ[j] + z[s] >= fit.ϵ[j, s])
            @constraint(lp_model, w0, 
                λ[1] == 0)
            @constraint(lp_model, subDiff,
                sum(fit.ccp .* λ) + sum(z) / S == fit.∂Gstar)

            # Define the objective function
            @objective(lp_model, Min, sign_y * f_y' * λ )

            # Surpress output and solve the linear program
            MOI.set(lp_model, MOI.Silent(), true)
            optimize!(lp_model)

            # Store bounds
            bounds[j, p] = -value.(lp_model[:λ])[j]
        end
    end

    # Return estimated bounds
    return bounds
end #GET_MTABOUNDS
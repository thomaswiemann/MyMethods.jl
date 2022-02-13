"""
myPQR(y, X, τ = 0.5, λ = 1; weights = nothing, control = nothing)

A simple implementation of penalized quantile regression. 
"""
struct myPQR 
    α::Array{Float64} # coefficients
    y::Array{Float64} # responses
    x # regressor
    τ # quantile
    λ # penalty parameter
    β # (Optional) coefficient on control variables
    control # (Optional) additional variables included in PQR
    lp_model # a JuMP model 
    weights # (Optional) vector of weights for loss function

    function myPQR(y, x, λ; τ = 0.5, weights = nothing, control = nothing)

        # Data parameters
        nobs = length(y)
        obs_order = sortperm(x[:])
        no_control = isnothing(control)

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

        # Define regression equation (w/ or w/o control)
        if no_control
            @constraint(lp_model, c[i = 1:nobs], 
                α[i] + uv[i, 1] - uv[i, 2] == y_o[i])
        else
            control_o = control[obs_order, :]
            n_control = size(control_o, 2)
            @variable(lp_model, β[1:n_control]) # coefficient on controls
            @constraint(lp_model, c[i = 1:nobs], 
                α[i] + control_o[i, :]' * β + uv[i, 1] - uv[i, 2] == y_o[i])
        end

        # Check for uniqueness of x_o. Define additional constraints if x_o is
        #     not unique.
        if unique(x_o) != nobs
            unq_x = unique(x_o)
            n_unq_x = length(unq_x)
            unq_indices = zeros(Int64, n_unq_x, 1)
            for j in 1:n_unq_x
                is_x_j = findall(x_o .== unq_x[j])
                unq_indices[j] = is_x_j[1]
                # Check whether value is no point mass
                if length(is_x_j) == 1
                    continue
                end
                # If value is a point mass, introduce new constraints
                @constraint(lp_model, [j = is_x_j[2:end]],
                    α[is_x_j[1]] == α[j])
            end
            # Set zero-valued h to 1 to avoid numerical errors.
            h[h .== 0] .= 1
        end

        # Define penalization terms
        @constraint(lp_model, penalty[i = 1:(nobs - 2)], 
            (α[i + 2] - α[i + 1]) / h[i + 1] - (α[i + 1] - α[i]) / h[i] == Δ[i])
        @constraint(lp_model, abs_penalty[i = 1:(nobs - 1), j = 0:1], 
            V[i] >= (-1)^j * Δ[i])

        # Define the check-loss function
        if isnothing(weights)
            weights_o = 1
        else 
            weights_o = weights[obs_order]
        end
        @objective(lp_model, Min, 
            sum(weights_o .* (τ * uv[1:nobs, 1] + (1 - τ) * uv[1:nobs, 2])) + 
            λ * sum(V))

        # Surpress output and solve the linear program
        MOI.set(lp_model, MOI.Silent(), true)
        optimize!(lp_model)

        # Set α
        if unique(x_o) == nobs
            α = value.(lp_model[:α])
        else
            α = value.(lp_model[:α])[unq_indices]
        end
        

        # (Optional) Set β
        if no_control
            β = nothing
        else
            β = value.(lp_model[:β])
        end

        # Define output
        new(α, y_o, x_o, τ, λ, β, control_o, lp_model, weights_o)
    end #MYPQR
end #MYPQR

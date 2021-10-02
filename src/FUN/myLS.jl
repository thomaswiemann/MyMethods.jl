"""
myLS(y, X)

A simple least squares implementation.
"""
struct myLS <: myEstimators
    β::Array{Float64} # coefficient
    y::Array{Float64} # response
    X::Array{Float64} # features

    # Define constructor function
    function myLS(y::Array{Float64}, X::Array{Float64})

    # Calculate LS
    β = (X' * X) \ (X' * y)

    # Organize and return output
    new(β, y, X)
    end #MYLS
end #MYLS

# Methods ======================================================================
"""
predict(fit::myLS; data)

A method to calculate predictions of a myLS object.
"""
function predict(fit::myLS, data::Array{Float64})
    # Return predicted values
    return data * fit.β
end #PREDICT.MYLS

"""
inference(fit::myLS; heteroskedastic, print_df)

A method to calculate standard errors of a myLS object.

To do: clustered se
"""
function inference(fit::myLS; heteroskedastic::Bool=false, print_df::Bool=true)
    # Obtain data parameters
    N = length(fit.y)
    K = size(fit.X, 2)

    # Calculate covariance matrix and standard errors
    u = fit.y - predict(fit) # residuals
    # Covariance for LS
    XX_inv = inv(fit.X' * fit.X)
    if !heteroskedastic
        # homoskedastic se
        covar = sum(u.^2) * XX_inv
	    covar = covar .* (1 / (N - K)) # dof adjustment
    else
        # heteroskedastic se
        covar = XX_inv * ((fit.X .* (u.^2))' * fit.X) * XX_inv
		covar = covar .* (N / (N - K)) # dof adjustment
    end
    se = sqrt.(covar[diagind(covar)])

    # Calculate t-statistic and p-values
    t_stat = fit.β ./ se
    p_val = 2 * cdf.(Normal(), -abs.(t_stat))

    # Print estimates
    if print_df
        out_df = DataFrame(hcat(fit.β, se, t_stat, p_val), :auto)
        rename!(out_df, ["coef", "se", "t-stat", "p-val"])
        display(out_df)
    end
    # Organize and return output
    output = (β = fit.β, se = se, t = t_stat, p = p_val)
    return output
end #INFERENCE.MYLS

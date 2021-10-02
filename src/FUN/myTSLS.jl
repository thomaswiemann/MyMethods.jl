"""
myTSLS(y, D, instrument, control)

A simple two stage least squares implementation.
"""
struct myTSLS <: myEstimators
    β::Array{Float64} # coefficient
    y::Array{Float64} # response
	Z::Array{Float64} # combined first stage variables
    X::Array{Float64} # combined second stage variables
	FS::Array{Float64} # first stage coefficients

	# Define constructor function
	function myTSLS(y, D, instrument, control = nothing)
        # Add constant if no control is passed
        if isnothing(control) control = ones(length(y)) end

		# Define data matrices
		Z = hcat(control, instrument) # combined first stage variables
		X = hcat(D, control) # combined second stage variables

		# Calculate matrix products
		ZZ = Z' * Z
		DZ = X' * Z
		Zy = Z' * y
		FS = inv(ZZ) * DZ'

		# Calculate TSLS coefficient
		β = (DZ * FS)' \ (FS' * Zy)

		# Return output
		new(β, y, Z, X, FS)
	end #MYTSLS
end #MYTSLS

# Methods ======================================================================
"""
predict(fit::myTSLS; data)

A method to calculate predictions of a myTSLS object.
"""
function predict(fit::myTSLS, data::Array{Float64})
    # Return predicted values
    return data * fit.β
end #PREDICT.MYTSLS

"""
inference(fit::myLS; heteroskedastic, print_df)

A method to calculate standard errors of a myLS object.

To do: clustered se
"""
function inference(fit::myTSLS; heteroskedastic::Bool=false, cluster=nothing,
    print_df::Bool=true)
    # Obtain data parameters
    N = length(fit.y)

	# Define sample matrices
	Z = fit.Z; Kz = size(Z, 2)
	X = fit.X; Kx = size(X, 2)
	u = fit.y - predict(fit) # residuals

	# Calculate matrix products
	PZ = Z * fit.FS
	PZZPinv = inv(PZ' * PZ)

	# Calculate covariance matrix
	if !heteroskedastic
		# Covariance under homoskedasticity
		covar = sum(u.^2) .* PZZPinv ./ (N - Kz)
	else
		# Covariance under heteroskedasticity
		PZuuZP = ((PZ .* (u.^2))' * PZ) .* (N / (N - Kz))
		covar = PZZPinv * PZuuZP * PZZPinv
	end

	# Get standard errors
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
end #INFERENCE.MYTSLS

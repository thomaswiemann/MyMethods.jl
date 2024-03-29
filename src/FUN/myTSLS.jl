"""
myTSLS(y, D, instrument, control)

A simple two stage least squares implementation.
"""
struct myTSLS <: myEstimator
    β::Array{Float64} # coefficient
    y::Array{Float64} # response
	Z::Array{Float64} # combined first stage variables
    X::Array{Float64} # combined second stage variables
	FS::Array{Float64} # first stage coefficients

	# Define constructor function
	function myTSLS(y::Array{Float64}, D::Array{Float64},
                    instrument::Array{Float64}, control = nothing)
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
inference(fit::myLS; heteroskedastic, print_df)

A method to calculate standard errors of a myLS object.

To do: clustered se
"""
function inference(fit::myTSLS; heteroskedastic::Bool=false, cluster=nothing,
    print_df::Bool=true)
    # Obtain data parameters
    N = length(fit.y)
    Kx = size(fit.X, 2)
    Kz = size(fit.Z, 2)

	# Calculate matrix products
	PZ = fit.Z * fit.FS
	PZZPinv = inv(PZ' * PZ)

	# Calculate covariance matrix
    u = fit.y - predict(fit) # residuals
	if !heteroskedastic
		# Covariance under homoskedasticity
		covar = sum(u.^2) .* PZZPinv ./ (N - Kz)
	else
		# Covariance under heteroskedasticity
		PZuuZP = ((PZ .* (u.^2))' * PZ) .* (N / (N - Kz))
		covar = PZZPinv * PZuuZP * PZZPinv
	end

	# Get standard errors, t-statistics and p-values
    se = sqrt.(covar[diagind(covar)])
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

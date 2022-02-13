"""
myLQR(y, x, x0, K, h, τ = 0.5; kernel = "Epanechnikov", control = nothing)

A local quantile regression implementation.
"""
struct myLQR <: myEstimator
    β # coefficients
    y # response
    X # matrix of regressors
    x # running variable
    x0 # value of x at which to evaluate the kernel
    K # degree of local linear regression
	h # bandwidth
    kernel # kernel function
	control # additional variables included in LLR
    τ # quantile
    lp_model # a JuMP model 
    
    # Define constructor function
    function myLQR(y, x, x0, K, h, τ = 0.5; 
        kernel = "Epanechnikov", control = nothing)	

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
        qr_fit = myQR(y, X, τ, weights = w)
        β = qr_fit.β
        lp_model = qr_fit.lp_model

        # Organize and return output
        new(β, y, X, x, x0, K, h, kernel, control, τ, lp_model)
    end #MYLQR
end #MYLQR
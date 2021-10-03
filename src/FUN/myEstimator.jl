"""
myEstimator

An abstract type for estimators.
"""
abstract type myEstimator end

# Methods ======================================================================
"""
coef(fit::myEstimator)

A method to retrieve the coefficient from a myEstimator object.
"""
function coef(fit::myEstimator)
    return fit.β
end #COEF.MYESTIMATOR

"""
predict(fit::myEstimator)

A method to calculate predictions of a myEstimator object.
"""
function predict(fit::myEstimator, data = nothing)
  # Calculate and return predictions
  isnothing(data) ? fitted = fit.X * fit.β : fitted = data * fit.β
  return(fitted)
end #PREDICT.MYESTIMATOR

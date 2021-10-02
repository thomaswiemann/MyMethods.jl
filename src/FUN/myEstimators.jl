"""
myEstimators

An abstract type for estimators.
"""
abstract type myEstimators end

# Methods ======================================================================
"""
coef(fit::myEstimators)

A method to retrieve the coefficient from a myEstimators object.
"""
function coef(fit::myEstimators)
    return fit.β
end #COEF.MYESTIMATORS

"""
predict(fit::myEstimators)

A method to calculate predictions of a myEstimators object.
"""
function predict(fit::myEstimators, data = nothing)
  # Calculate and return predictions
  isnothing(data) ? fitted = fit.X * fit.β : fitted = data * fit.β
  return(fitted)
end #PREDICT.MYESTIMATORS

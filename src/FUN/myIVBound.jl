"""
myIVBound(y, x)

An implementation of the monotone IV bounds of Manski and Pepper (2000).
"""
struct myIVBound
    px
    Eyx
    unq_x

    function myIVBound(y, x)
        unq_x = sort(unique(x))
        px = zeros(length(unq_x))
        Eyx = zeros(length(unq_x))
        for u in 1:length(unq_x)
            # Estimate P(x = u) and E[y|x = u]
            indx = x .== unq_x[u]
            px[u] = sum(indx) / n
            Eyx[u] = mean(y[indx])
        end
        new(px, Eyx, unq_x)
    end #MYIVBOUND
end #MYIVBOUND


function coef(fit::myIVBound, s, t)
    Eyx_px = fit.Eyx .* fit.px

    from_t = fit.unq_x .> t
    until_s = fit.unq_x .< s
    
    ub = sum(Eyx_px[from_t]) + (fit.Eyx[fit.unq_x .== t] * sum(fit.px[.!from_t]))[1]
    lb = sum(Eyx_px[until_s]) + (fit.Eyx[fit.unq_x .== s] * sum(fit.px[.!until_s]))[1]
    
    return ub - lb
end


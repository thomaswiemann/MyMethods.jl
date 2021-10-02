using MyMethods
using Test

@testset "myLS.jl" begin
    # Generate example data
    n_i = 1000
    X = 2 * randn((n_i, 2))
    y = X * [1, -2] + randn((n_i, 1))

    # Estimate the least square regression
    ls_fit = myLS(y, X)

    # Check the methods
    β_hat = coef(ls_fit)
    y_hat = predict(ls_fit)
    se_hat = inference(ls_fit, print_df = false)
    se_hc_hat = inference(ls_fit, heteroskedastic = true, print_df = false)

    # Let's check that everything is of correct type.
    @test typeof(ls_fit) == myLS
    @test typeof(β_hat) == Array{Float64,2}
    @test typeof(y_hat) == Array{Float64,2}
    @test typeof(se_hat) == NamedTuple{(:β, :se, :t, :p),
                                       Tuple{Array{Float64,2},
                                             Array{Float64,1},
                                             Array{Float64,2},
                                             Array{Float64,2}}}
    @test typeof(se_hc_hat) == NamedTuple{(:β, :se, :t, :p),
                                          Tuple{Array{Float64,2},
                                                Array{Float64,1},
                                                Array{Float64,2},
                                                Array{Float64,2}}}
end

@testset "myTSLS.jl" begin
    # Generate example data
    n_i = 1000
    ν = randn((n_i, 1))
    instrument = 2 * randn((n_i, 2))
    D = instrument * [1, -2] + ν
    y = D + 0.3 * ν + randn((n_i, 1))

    # Estimate the least square regression
    tsls_fit = myTSLS(y, D, instrument)

    # Check the methods
    β_hat = coef(tsls_fit)
    y_hat = predict(tsls_fit)
    se_hat = inference(tsls_fit, print_df = false)
    se_hc_hat = inference(tsls_fit, heteroskedastic = true, print_df = false)

    # Let's check that everything is of correct type.
    @test typeof(tsls_fit) == myTSLS
    @test typeof(β_hat) == Array{Float64,2}
    @test typeof(y_hat) == Array{Float64,2}
    @test typeof(se_hat) == NamedTuple{(:β, :se, :t, :p),
                                       Tuple{Array{Float64,2},
                                             Array{Float64,1},
                                             Array{Float64,2},
                                             Array{Float64,2}}}
    @test typeof(se_hc_hat) == NamedTuple{(:β, :se, :t, :p),
                                          Tuple{Array{Float64,2},
                                                Array{Float64,1},
                                                Array{Float64,2},
                                                Array{Float64,2}}}
end

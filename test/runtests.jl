using MyMethods
using Test

@testset "myLS.jl" begin
    # Generate example data
    n_i = 1000
    X = 2 * randn((n_i, 2))
    y = X * [1, -2] + randn((n_i, 1))
    w = rand(n_i)
    w = w ./ sum(w)

    # Estimate the least square regression
    ls_fit = myLS(y, X, weights = w)

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

@testset "mySieve.jl" begin
      # Generate example data
      n_i = 1000
      x = randn(n_i)
      X = get_basis(x, "Bernstein", 5, nothing)
      y = X * randn(6) + randn((n_i, 1))
  
      # Estimate the sieve regression
      sieve_fit = mySieve(y, x; basis="Bernstein", K=5)

      # Check the methods
      β_hat = coef(sieve_fit)
      y_hat = predict(sieve_fit)

      # Let's check that everything is of correct type.
      @test typeof(sieve_fit) == mySieve
end

@testset "myQR.jl" begin
      # Generate example data
      n_i = 1000
      X = randn(n_i, 2)
      y = X * [0.5, -0.5] + randn((n_i, 1))
  
      # Estimate the quantile regression regression
      qr_fit = myQR(y, X, τ = 0.5)

      # Check the methods
      β_hat = coef(qr_fit)
      y_hat = predict(qr_fit)

      # Let's check that everything is of correct type.
      @test typeof(qr_fit) == myQR
end

@testset "myLLR.jl" begin
      # Generate example data
      n_i = 1000
      x = 2 * randn((n_i, 1))
      y = log.(x.^2) + randn((n_i, 1))
      w = rand(n_i)
      w = w ./ sum(w)
  
      # Estimate the least square regression
      llr_fit = myLLR(y, x, x[1], 2, 0.5)
  
      # Check the methods
      β_hat = coef(llr_fit)
      y_hat = predict(llr_fit)
  
      # Let's check that everything is of correct type.
      @test typeof(llr_fit) == myLLR
  end

  @testset "myLQR.jl" begin
      # Generate example data
      n_i = 1000
      x = 2 * randn((n_i, 1))
      y = log.(x.^2) + randn((n_i, 1))
  
      # Estimate the least square regression
      lqr_fit = myLQR(y, x, x[1], 2, 0.5)
  
      # Check the methods
      β_hat = coef(lqr_fit)
      y_hat = predict(lqr_fit)
  
      # Let's check that everything is of correct type.
      @test typeof(lqr_fit) == myLQR
  end

  @testset "myPQR.jl" begin
      # Generate example data
      n_i = 1000
      x = 2 * (randn((n_i, 1)) .<= 0) .+ 1
      X = randn((n_i, 3))
      y = log.(x.^2) + X * [1, 1, 1] + randn((n_i, 1))
  
      # Estimate the least square regression
      lqr_fit = myPQR(y, x, 1, control = X)

      # Check the methods
      α_hat = lqr_fit.α
      β_hat = lqr_fit.β
  
      # Let's check that everything is of correct type.
      @test typeof(lqr_fit) == myPQR
  end

  @testset "myMTA.jl" begin
      # Generate example data
      n_i = 1000
      J = 5
      X = randn((n_i, J - 1, 3))
      mu = zeros((n_i, J - 1))
      for j in 1:(J - 1)
            mu[:, j] = X[:, j, :] * [1, 1, 1]
      end
      mu = exp.(mu)
      CCP = mu ./ (1 .+ sum(mu, dims = 2))
      CCP = hcat(1 .- sum(CCP, dims = 2), CCP)

      



      wsample(collect(1:J), CCP[1, :], 5)
      y = mapslices(x -> wsample(collect(1:J), x), CCP, dims = 2)

      mu = X .* [1, 1, 1]
      CCP = exp.(mu)
      y = log.(x.^2) + X * [1, 1, 1] + randn((n_i, 1))
  
      # Estimate the least square regression
      lqr_fit = myPQR(y, x, 1, control = X)

      # Check the methods
      α_hat = lqr_fit.α
      β_hat = lqr_fit.β
  
      # Let's check that everything is of correct type.
      @test typeof(lqr_fit) == myPQR
  end
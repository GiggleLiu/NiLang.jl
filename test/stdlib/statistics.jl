import Statistics
using Test, Random
using NiLang, NiLang.AD
using Distributions

@testset "statistics" begin
    x = randn(100)
    @test i_mean_sum(0.0, 0.0, x)[1] ≈ Statistics.mean(x)
    @test all(i_var_mean_sum(0.0, 0.0, 0.0, 0.0, copy(x)) .≈ (Statistics.var(x), Statistics.var(x)*99, Statistics.mean(x), sum(x), x))
    @test all(isapprox.((~i_var_mean_sum)(i_var_mean_sum(0.0, 0.0, 0.0, 0.0, copy(x))...), (0.0, 0.0, 0.0, 0.0, x), atol=1e-8))
end

@testset "normal log pdf" begin
    out = 0.0
    x = 1.0
    μ = 0.3
    σ = 1.5
    l1 = i_normal_logpdf(out, x, μ, σ)
    distri = Normal(μ, σ)
    l2 = logpdf(distri, x)
    @test l1[1] ≈ l2
    @test check_inv(i_normal_logpdf, (out, x, μ, σ))
  end

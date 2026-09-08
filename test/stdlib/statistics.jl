import Statistics
using Test, Random
using NiLang, NiLang.AD
using Distributions

@testset "statistics" begin
    x = randn(100)
    y = randn(100)

    @test check_inv(i_mean_sum, (0.0, 0.0, x))
    @test all(i_mean_sum(0.0, 0.0, x) .≈ (Statistics.mean(x), sum(x), x))
    @test check_inv(i_sum, (0.0, x))
    @test i_sum(0.0, x)[1] ≈ sum(x)
    @test check_inv(i_mean, (0.0, x))
    @test i_mean(0.0, x)[1] ≈ mean(x)

    info = VarianceInfo(Float64)
    @test almost_same(i_var_mean_sum(info, copy(x))[1], VarianceInfo(Statistics.var(x), Statistics.var(x)*99, Statistics.mean(x), sum(x)))
    @test almost_same((~i_var_mean_sum)(i_var_mean_sum(info, copy(x))...), (info, x))
    @test almost_same(i_cor_cov(0.0, 0.0, copy(x), copy(y)), (Statistics.cor(x,y), Statistics.cov(x,y), x, y))
    @test almost_same((~i_cor_cov)(i_cor_cov(0.0, 0.0, copy(x), copy(y))...), (0.0, 0.0, x, y))
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

import Statistics
using Test, Random
using NiLang, NiLang.AD
using Distributions

@testset "statistics" begin
    x = randn(100)
    @test NiFunctions.mean(0.0, x)[1] ≈ Statistics.mean(x)
    @test all(NiFunctions.var_and_mean(0.0, 0.0, copy(x)) .≈ (Statistics.var(x), Statistics.mean(x), x))
    @test all(isapprox.((~NiFunctions.var_and_mean)(NiFunctions.var_and_mean(0.0, 0.0, copy(x))...), (0.0, 0.0, x), atol=1e-8))
end

@testset "normal log pdf" begin
    out = 0.0
    x = 1.0
    μ = 0.3
    σ = 1.5
    l1 = NiFunctions.normal_logpdf(out, x, μ, σ)
    distri = Normal(μ, σ)
    l2 = logpdf(distri, x)
    @test l1[1] ≈ l2
    @test check_inv(NiFunctions.normal_logpdf, (out, x, μ, σ))
    x2 = [0.2, 0.9]
    μ2 = [0.0, 0.2]
    σ = 1.0
    distri = MultivariateNormal(μ2, [σ 0; 0 σ])
    @test check_inv(NiFunctions.normal_logpdf2d, (out, x2, μ2, σ))
    @test logpdf(distri, x2) ≈ NiFunctions.normal_logpdf2d(0.0, x2, μ2, σ)[1]
end




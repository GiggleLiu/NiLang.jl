include("nnlib.jl")
using Test, Random
using Distributions, StatsBase, LinearAlgebra

@testset "leapfrog" begin
    function f(F, y0, N, dt)
        y = y0
        for i=1:N
           y += F(y)*dt
        end
        return y
    end
    x = 0.8
    @instr LeapFrog(⊕(sin))(x, (); Nt=10000, dt=0.0001)
    truth = f(sin, 0.8, 10000, 0.0001)
    @test isapprox(x, truth; atol=1e-3)
end

@testset "normal log pdf" begin
    out = 0.0
    x = 1.0
    μ = 0.3
    σ = 1.5
    l1 = normal_logpdf(out, x, μ, σ)
    distri = Normal(μ, σ)
    l2 = logpdf(distri, x)
    @test l1[1] ≈ l2
    @test check_inv(normal_logpdf, (out, x, μ, σ))
    x2 = [0.2, 0.9]
    μ2 = [0.0, 0.2]
    σ = 1.0
    distri = MultivariateNormal(μ2, [σ 0; 0 σ])
    @test check_inv(normal_logpdf2d, (out, x2, μ2, σ))
    @test logpdf(distri, x2) ≈ normal_logpdf2d(0.0, x2, μ2, σ)[1]
end

#=
@testset "neural ode" begin
    Random.seed!(2)
    θ = 0.5
    nsample = 100
    Nt = 50
    dt = 0.01

    source_distri = Normal(1, 0.5)
    xs0 = rand(source_distri, nsample)
    logp0 = max.(logpdf.(source_distri, xs0), -1000)

    # solving the ode, and obtain the probability change
    v_xs = copy(xs0)
    v_ks = copy(xs0)
    v_logp = copy(logp0)
    v_θ = θ
    field_out = 0.0
    tape = ode_with_logp!(v_xs, v_ks, v_logp, field, field_out, v_θ, Nt, dt)
    @test check_inv(tape; verbose=true, atol=1e-3)
end
=#

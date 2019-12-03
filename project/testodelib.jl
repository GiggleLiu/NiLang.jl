include("nnlib.jl")
include("fields.jl")
using Test, Random
using Distributions, StatsBase, LinearAlgebra

@testset "field" begin
    lf = LinearField(0.5)
    out, x = 0.0, 2.0
    @instr get_field(lf, out, x)
    @test out == 1.0

    x = 2.0
    y = 0.5
    θ = 0.5
    @instr update_field(LinearField(θ), y, x; dt=0.1)
    @test y === 0.6
    @test check_inv(update_field, (LinearField(θ), y, x); kwargs=Dict(:dt=>0.1))

    x = 2.0
    y = PVar(0.5)
    θ = 0.5
    @instr update_field(LinearField(θ), y, x; dt=0.1)
    @test val(y) === 0.6
    @test y.logp ≈ -0.05
end

@i function get_field(ps::typeof(⊕(sin)), field_out, y)
    field_out += sin(y)
end

function f(F, y0, N, dt)
    y = y0
    for i=1:N
       y += F(y)*dt
    end
    return y
end

@testset "leapfrog" begin
    x = 0.8
    truth = f(sin, 0.8, 10000, 0.0001)
    field = ⊕(sin)
    @instr Dup(x)
    @instr leapfrog(field, x; Nt=10000, dt=0.0001)
    @test isapprox(x, truth; atol=1e-3)

    # integrate sin(θ), get -log(det(∂cos(θ)/∂θ))
    lf = LinearField(0.5)
    x = PVar(0.8)
    truth = val(x)
    for i=1:1000
        truth += 0.001*truth*lf.θ
    end
    @instr Dup(x)
    @instr leapfrog(lf, x; Nt=1000, dt=0.001)
    @test isapprox(val(x), truth; atol=1e-3)


    @test isapprox(x.x.logp, -0.5, atol=1e03) # TODO: manual check

    lf = LinearField(0.5)
    xs = randn(10)
    μ, σ = 0.0, 1.0
    loss_out = 0.0
    @instr Dup.(xs)
    @instr ode_loss(μ, σ, lf, xs, loss_out; Nt=100, dt=0.01)
    @show loss_out
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

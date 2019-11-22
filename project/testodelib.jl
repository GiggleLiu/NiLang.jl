include("odelib.jl")
using Test, Random

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
    @newvar v_xs = copy(xs0)
    @newvar v_ks = copy(xs0)
    @newvar v_logp = copy(logp0)
    @newvar v_θ = θ
    @newvar field_out = 0.0
    tape = ode_with_logp!(v_xs, v_ks, v_logp, field, field_out, v_θ, Nt, dt)
    resetreg(tape)
    @test check_inv(tape; verbose=true, atol=1e-3)
end

@testset "logpdf" begin
    @newvar out = 0.0
    tape = normal_logpdf(out, 2.3, 1.0, 1.0)
    play!(tape)
    out[] == logpdf(Normal(1.0, 1.0), 2.3)
end

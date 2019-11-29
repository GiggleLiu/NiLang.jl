using NiLang, NiLang.AD
using Test, Random

@testset "softmax_crossentropy" begin
    Random.seed!(2)
    x = randn(10)
    x0 = copy(x)
    p = randn(10); p=p./maximum(p)
    res = _sce(x, p)
    imax = 0
    Z = 0.0
    out = 0.0
    xmax = 0.0
    x_ = x
    p_ = p
    @instr softmax_cross_entropy(x_, p_, imax, xmax, Z, out)
    @show Z
    @test isapprox(imax, argmax(x0), atol=1e-8)
    @test isapprox(out, res[], atol=1e-8)
    @instr (~softmax_cross_entropy)(x_, p_, imax, xmax, Z, out)
    args = x_, p_, imax, xmax, Z, out
    @test check_inv(softmax_cross_entropy, args)
    args = x_, p_, imax, xmax, Z, Loss(out)
    @test check_grad(softmax_cross_entropy, args; verbose=true)
end

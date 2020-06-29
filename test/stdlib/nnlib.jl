using Test, Random
using NiLang, NiLang.AD

function _sce(x::AbstractArray{T,N}, p) where {T,N}
    x = x .- maximum(x; dims=N)  # avoid data overflow
    rho = exp.(x)
    Z = sum(rho; dims=N)
    return dropdims(sum((log.(Z) .- x) .* p; dims=N), dims=N)
end


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
    @instr i_softmax_crossentropy(x_, p_, imax, xmax, Z, out)
    @show Z
    @test isapprox(imax, argmax(x0), atol=1e-8)
    @test isapprox(out, res[], atol=1e-8)
    @instr (~i_softmax_crossentropy)(x_, p_, imax, xmax, Z, out)
    args = x_, p_, imax, xmax, Z, out
    @test check_inv(i_softmax_crossentropy, args)
    args = x_, p_, imax, xmax, Z, out
    @test check_grad(i_softmax_crossentropy, args; iloss=6, verbose=true)
end

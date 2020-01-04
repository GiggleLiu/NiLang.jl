using NiLang, NiLang.AD

@i function iexp(out!, x::T; atol::Float64=1e-14) where T
    @anc anc1 = zero(T)
    @anc anc2 = zero(T)
    @anc anc3 = zero(T)
    @anc iplus = zero(T)

    out! ⊕ 1.0
    anc1 ⊕ 1.0
    while (value(anc1) > atol, !isapprox(iplus, 0.0))
        iplus ⊕ 1.0
        anc2 += anc1 * x
        anc3 += anc2 / iplus
        out! ⊕ anc3
        # speudo inverse
        anc1 -= anc2 / x
        anc2 -= anc3 * iplus
        SWAP(anc1, anc3)
    end

    ~(while (value(anc1) > atol, !isapprox(iplus, 0.0))
        iplus ⊕ 1.0
        anc2 += anc1 * x
        anc3 += anc2 / iplus
        # speudo inverse
        anc1 -= anc2 / x
        anc2 -= anc3 * iplus
        SWAP(anc1, anc3)
    end)
    anc1 ⊖ 1.0
end

using Test
# NOTE: to allow high performance use of f += T(x).
# Now, this kind of use is a performance killer.
@testset "iexp" begin
    out = 0.0
    x = 1.3
    @instr iexp(out, x)
    res = exp(x)
    @test check_inv(iexp,(out, x))
    @test out ≈ res

    out = 0.0
    x = 1e-9
    @instr iexp(out, x)
    res = exp(x)
    @test check_inv(iexp,(out, x))
    @test out ≈ res

    out = 0.0
    x = 1.0

    @instr iexp(out, x)
    res = exp(x)
    @test check_inv(iexp,(out, x))
    @test out ≈ res
end

@testset "iexp grad" begin
    out = 0.0
    x = 1.6
    gres = exp(x)
    @test check_inv(iexp, (out, x); verbose=true)
    @test check_grad(iexp, (Loss(out), x); verbose=true)

    out = 0.0
    x = 1.6
    @instr iexp'(Loss(out), x)
    @test grad(x) ≈ gres

    h = taylor_hessian(iexp, (Loss(0.0), 1.6))
    nh = nhessian(iexp, (Loss(0.0), 1.6))
    @test h ≈ nh
end

Base.zero(x::Type{HessianData{T}}) where T = zero(T)

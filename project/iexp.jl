using NiLang, NiLang.AD

@i function iexp(out!, x::T; atol::Float64=1e-14) where T
    @anc anc1 = zero(T)
    @anc anc2 = zero(T)
    @anc anc3 = zero(T)
    @anc iplus = 0
    @anc expout = zero(T)

    out! ⊕ 1.0
    @routine r1 begin
        anc1 ⊕ 1.0
        while (value(anc1) > atol, iplus != 0)
            iplus ⊕ 1
            anc2 += anc1 * x
            anc3 += anc2 / iplus
            expout ⊕ anc3
            # speudo inverse
            anc1 -= anc2 / x
            anc2 -= anc3 * iplus
            SWAP(anc1, anc3)
        end
    end

    out! ⊕ expout

    ~@routine r1
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
    NiLangCore.GLOBAL_ATOL[] = 1e-2
    out = 0.0
    x = 1.6
    gres = exp(x)
    @test check_inv(iexp, (out, x); verbose=true)
    @test check_grad(iexp, (Loss(out), x); verbose=true)

    out = 0.0
    x = 1.6
    @instr iexp'(Loss(out), x)
    @test grad(x) ≈ gres

    h1 = (iexp''(Loss(0.0), 1.6); collect_hessian())
    h2 = simple_hessian(iexp, (Loss(0.0), 1.6))
    nh = nhessian(iexp, (Loss(0.0), 1.6))
    @show h1, h2
    @test isapprox(h1, nh, atol=1e-3)
    @test isapprox(h2, nh, atol=1e-3)
end

using NiLang, NiLang.AD

@i function iexp(out!, x::T; niter::Int=30) where T
    @anc anc1::T
    @anc anc2::T
    @anc anc3::T
    @anc iplus::T

    out! += T(1.0)
    if (val(x) != 0.0, val(out!)!=1.0)
        anc1 += T(1.0)
        for i=1:niter
            iplus += T(1.0)
            anc2 += anc1 * x
            anc3 += anc2 / iplus
            out! += anc3
            # speudo inverse
            anc1 -= anc2 / x
            anc2 -= anc3 * iplus
            SWAP(anc1, anc3)
        end

        ~(for i=1:niter
            iplus += T(1.0)
            anc2 += anc1 * x
            anc3 += anc2 / iplus
            # speudo inverse
            anc1 -= anc2 / x
            anc2 -= anc3 * iplus
            SWAP(anc1, anc3)
        end)
        anc1 -= T(1.0)
    end
end

using Test
@testset "iexp" begin
    out = 0.0
    x = 1.0
    @instr iexp(out, x; niter=30)
    res = exp(x)
    @test check_inv(iexp,(out, x))
    @test isapprox(out, res, atol=0.01)
end

@testset "iexp grad" begin
    out = 0.0
    x = 1.6
    gres = exp(x)
    @test check_inv(iexp, (Loss(out), x); kwargs=(:niter=>20,), verbose=true)
    @test check_grad(iexp, (Loss(out), x); kwargs=(:niter=>20,), verbose=true)

    out = 0.0
    x = 1.6
    @instr iexp'(Loss(out), x; niter=20)
    @test grad(x) ≈ gres
end

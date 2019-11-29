using NiLang, NiLang.AD

@i function isqrt(out!, x::T; atol::Float64=1e-8) where T
    @anc anc1::T
    @anc anc2::T
    @anc yn::T
    @anc ynpre::T
    @anc diff::T
    @anc count::Int

    @routine forward begin
        yn += x
        while (!isapprox(val(yn)^2, val(x); atol=atol) && count<50, count!=0)
            anc1 += x / yn
            anc1 -= yn
            yn += anc1 / 2
            # speudo inverse
            @routine getynpre begin
                @safe @show anc2
                if (val(x) > 1.0, ~)
                    anc2 += yn^2
                    anc2 -= x
                    ynpre += anc2^0.5  # +-
                else
                    anc2 -= yn^2
                    anc2 += x
                    ynpre -= anc2^0.5  # +-
                end
                ynpre += yn
                @safe @show ynpre, yn, anc2
            end
            @safe @show ynpre
            anc1 += ynpre
            anc1 -= x/ynpre
            ## uncompute ynpre and anc2
            ~@routine getynpre
            count += 1
        end
    end
    out! += yn

    ~@routine forward
end

using Test
# NOTE: to allow high performance use of f += T(x).
# Now, this kind of use is a performance killer.
@testset "isqrt" begin
    out = 0.0
    x = 1.3
    @instr isqrt(out, x)
    res = sqrt(x)
    @test check_inv(isqrt,(out, x))
    @test isapprox(out, res; atol=1e-8)

    @show "@"
    out = 0.0
    x = 0.3
    @instr isqrt(out, x)
    res = sqrt(x)
    @test check_inv(isqrt,(out, x); verbose=true)
    @test isapprox(out, res; atol=1e-8)

    out = 0.0
    x = 1.0

    @instr isqrt(out, x)
    res = sqrt(x)
    @test check_inv(isqrt,(out, x))
    @test out ≈ res
end

@testset "isqrt grad" begin
    out = 0.0
    x = 1.6
    gres = sqrt(x)
    @test check_inv(isqrt, (Loss(out), x); verbose=true)
    @test check_grad(isqrt, (Loss(out), x); verbose=true)

    out = 0.0
    x = 1.6
    @instr isqrt'(Loss(out), x)
    @test grad(x) ≈ gres
end

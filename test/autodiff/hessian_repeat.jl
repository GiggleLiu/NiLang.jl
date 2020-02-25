using NiLang, NiLang.AD, Test
using NiLang.AD: hessian_numeric

@testset "hessian" begin
    h1 = hessian_repeat(⊕(*), (Loss(0.0), 2.0, 3.0))
    h2 = hessian_numeric(⊕(*), (Loss(0.0), 2.0, 3.0))
    @test h1 ≈ h2

    @i function test(a,b,c,d)
        a += b*c
        a += b^d
        c += b/d
        ROT(a, c, d)
        b += d * d
        a += c * d
    end
    h1 = hessian_repeat(test, (Loss(0.0), 2.0, 1.0, 3.0))
    h2 = hessian_numeric(test, (Loss(0.0), 2.0, 1.0, 3.0))
    @show h2
    @test isapprox(h1, h2, atol=1e-8)
end

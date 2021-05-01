using NiLang, NiLang.AD, Test
using NiLang.AD: hessian_numeric

@testset "hessian" begin
    h1 = hessian_backback(PlusEq(*), (0.0, 2.0, 3.0); iloss=1)
    h2 = hessian_numeric(PlusEq(*), (0.0, 2.0, 3.0); iloss=1)
    @test h1 ≈ h2

    @i function test(a,b,c,d)
        a += b*c
        a += b^d
        c += b/d
        ROT(a, c, d)
        b += d ^ 2
        a += c * d
    end
    h1 = hessian_backback(test, (0.0, 2.0, 1.0, 3.0); iloss=1)
    h2 = hessian_numeric(test, (0.0, 2.0, 1.0, 3.0); iloss=1)
    @show h2
    @test isapprox(h1, h2, atol=1e-8)
end

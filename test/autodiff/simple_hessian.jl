using NiLang, NiLang.AD, Test

@testset "hessian" begin
    h1 = simple_hessian(⊕(*), (Loss(0.0), 2.0, 3.0))
    h2 = nhessian(⊕(*), (Loss(0.0), 2.0, 3.0))
    @test h1 ≈ h2

    @i function test(a,b,c,d)
        a += b*c
        a += b^d
        ROT(a, c, d)
    end
    h1 = simple_hessian(test, (Loss(0.0), 2.0, 1.0, 3.0))
    h2 = nhessian(test, (Loss(0.0), 2.0, 1.0, 3.0))
    @show h2
    @test h1 ≈ h2
end

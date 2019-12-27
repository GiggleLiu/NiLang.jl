using NiLang, NiLang.AD, Test

@testset "hessian" begin
    h1 = simple_hessian(⊕(*), (Loss(0.0), 2.0, 3.0))
    h2 = nhessian(⊕(*), (Loss(0.0), 2.0, 3.0))
    @test h1 ≈ h2
end

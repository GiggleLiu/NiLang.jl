using NiLang, NiLang.AD
using Test

@testset "check grad" begin
    @test check_grad(⊕, (Loss(1.0), 2.0))
    @test check_grad(⊖, (Loss(1.0), 2.0))
    @test check_grad(⊕(*), (Loss(1.0), 2.0, 2.0); verbose=true)
    @test check_grad(⊖(*), (Loss(1.0), 2.0, 2.0); verbose=true)
    @test check_grad(NEG, (Loss(1.0),); verbose=true)
    @test check_grad(⊕(/), (Loss(1.0), 2.0, 2.0); verbose=true)
    @test check_grad(⊖(/), (Loss(1.0), 2.0, 2.0); verbose=true)
    @test check_grad(ROT, (Loss(1.0), 2.0, 2.0); verbose=true)
    @test check_grad(ROT, (1.0, Loss(2.0), 2.0); verbose=true)
    @test check_grad(IROT, (Loss(1.0), 2.0, 2.0); verbose=true)
    @test check_grad(IROT, (1.0, Loss(2.0), 2.0); verbose=true)
end

using NiLang, NiLang.AD
using Test

@testset "check grad" begin
    for opm in [⊕, ⊖]
        @test check_grad(opm, (Loss(1.0), 2.0); verbose=true)
        @test check_grad(opm(*), (Loss(1.0), 2.0, 2.0); verbose=true)
        @test check_grad(opm(^), (Loss(1.0), 2.0, 2); verbose=true)
        @test check_grad(opm(^), (Loss(1.0), 2.0, 2.0); verbose=true)
        @test check_grad(opm(exp), (Loss(1.0), 2.0); verbose=true)
        @test check_grad(opm(log), (Loss(1.0), 2.0); verbose=true)
        @test check_grad(opm(sin), (Loss(1.0), 2.0); verbose=true)
        @test check_grad(opm(cos), (Loss(1.0), 2.0); verbose=true)
        @test check_grad(opm(/), (Loss(1.0), 2.0, 2.0); verbose=true)
    end
    @test check_grad(NEG, (Loss(1.0),); verbose=true)
    @test check_grad(ROT, (Loss(1.0), 2.0, 2.0); verbose=true)
    @test check_grad(ROT, (1.0, Loss(2.0), 2.0); verbose=true)
    @test check_grad(IROT, (Loss(1.0), 2.0, 2.0); verbose=true)
    @test check_grad(IROT, (1.0, Loss(2.0), 2.0); verbose=true)
end

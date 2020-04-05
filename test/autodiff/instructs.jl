using NiLang, NiLang.AD
using Test

@testset "check grad" begin
    for opm in [⊕, ⊖]
        @test check_grad(opm(identity), (1.0, 2.0); verbose=true, iloss=1)
        @test check_grad(opm(*), (1.0, 2.0, 2.0); verbose=true, iloss=1)
        @test check_grad(opm(+), (1.0, 2.0, 2.0); verbose=true, iloss=1)
        @test check_grad(opm(-), (1.0, 2.0, 2.0); verbose=true, iloss=1)
        @test check_grad(opm(^), (1.0, 2.0, 2); verbose=true, iloss=1)
        @test check_grad(opm(^), (1.0, 2.0, 2.0); verbose=true, iloss=1)
        @test check_grad(opm(abs), (1.0, -2.0); verbose=true, iloss=1)
        @test check_grad(opm(exp), (1.0, 2.0); verbose=true, iloss=1)
        @test check_grad(opm(log), (1.0, 2.0); verbose=true, iloss=1)
        @test check_grad(opm(sin), (1.0, 2.0); verbose=true, iloss=1)
        @test check_grad(opm(cos), (1.0, 2.0); verbose=true, iloss=1)
        @test check_grad(opm(/), (1.0, 2.0, 2.0); verbose=true, iloss=1)
    end
    @test check_grad(NEG, (1.0,); verbose=true, iloss=1)
    @test check_grad(mulint, (1.5, 2); verbose=true, iloss=1)
    @test check_grad(divint, (1.5, 2); verbose=true, iloss=1)
    @test check_grad(ROT, (1.0, 2.0, 2.0); verbose=true, iloss=1)
    @test check_grad(ROT, (1.0, 2.0, 2.0); verbose=true, iloss=2)
    @test check_grad(IROT, (1.0, 2.0, 2.0); verbose=true, iloss=1)
    @test check_grad(IROT, (1.0, 2.0, 2.0); verbose=true, iloss=2)
end

@testset "AD over pop" begin
    @i function mean(out!, x)
        anc ← zero(out!)
        for i=1:length(x)
            anc += identity(x[i])
        end
        out! += anc / length(x)
        ipush!(anc)
    end

    @test check_grad(mean, (0.0, [1,2,3.0, 4.0]); iloss=1)
end

using NiLang, NiLang.AD
using Test

@testset "check grad" begin
    @test check_grad(⊕, (Loss(1.0), 2.0))
    @test check_grad(⊖, (Loss(1.0), 2.0))
    @test check_grad(⊕(*), (Loss(1.0), 2.0, 2.0); verbose=true)
    @test check_grad(⊖(*), (Loss(1.0), 2.0, 2.0); verbose=true)
    @test check_grad(NEG, (Loss(1.0),); verbose=true)
    #@test check_grad(⊕(/), (Loss(1.0), 2.0, 2.0); verbose=true)
    #@test check_grad(⊖(/), (Loss(1.0), 2.0, 2.0); verbose=true)
end

@testset "neg sign" begin
    @i function test(out, x, y)
        out += x * (-y)
    end
    @test check_inv(test, (Loss(0.1), 2.0, -2.5); verbose=true)
    @test check_grad(test, (Loss(0.1), 2.0, -2.5); verbose=true)
end

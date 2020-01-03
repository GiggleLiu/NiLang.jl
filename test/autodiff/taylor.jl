using NiLang, NiLang.AD
using NiLang.AD: hrow, hcol, size_paramspace
using Test

@testset "HessianData" begin
    hdata = zeros(3,3)
    gdata = [1.0, 0, 0]
    out! = HessianData(6.0, gdata, hdata, 1)
    a = HessianData(2.0, gdata, hdata, 2)
    b = HessianData(3.0, gdata, hdata, 3)

    @test size_paramspace(out!) == 3

    @test chfield(out!, Val(:x), 0.5) == HessianData(0.5, gdata, hdata, 1)
    @test chfield(out!, value, 0.6) == HessianData(0.6, gdata, hdata, 1)
    chfield(a, grad, 0.5)
    @test gdata == [1.0,0.5,0.0]
    chfield(b, hrow, [1,2,3.0])
    @test hdata == [0 0 0; 0 0 0; 1.0 2.0 3.0]
    chfield(b, hcol, [1,2,3.0])
    @test hdata == [0 0 1; 0 0 2; 1 2 3.0]
end

@testset "hessian /" begin
    hdata = zeros(3,3)
    gdata = [1.0, 0, 0.0]
    out! = HessianData(0.0, gdata, hdata, 1)
    a = HessianData(0.5, gdata, hdata, 2)
    b = HessianData(0.2, gdata, hdata, 3)
    @instr ⊖(/)(out!, a, b)
    @test out!.x == -2.5
    @test out!.gradient == [1, 5.0, -12.5]
    @test out!.hessian == [0 0 0; 0 0 -25; 0 -25 125.0]
end

@testset "hessian ^" begin
    hdata = zeros(3,3)
    gdata = [1.0, 0, 0]
    out! = HessianData(0.0, gdata, hdata, 1)
    a = HessianData(2.0, gdata, hdata, 2)
    b = HessianData(3.0, gdata, hdata, 3)
    @instr ⊖(^)(out!, a, b)
    @test out!.x == -8
    @test out!.gradient == [1, 12.0, log(2.0)*8]
    @test out!.hessian == [0 0 0; 0 12 4+4*3*log(2.0); 0 4+4*3*log(2.0) 8*log(2.0)^2]
end

@testset "hessian *" begin
    hdata = zeros(3,3)
    gdata = [1.0, 0, 0]
    out! = HessianData(6.0, gdata, hdata, 1)
    a = HessianData(2.0, gdata, hdata, 2)
    b = HessianData(3.0, gdata, hdata, 3)
    @instr ⊖(*)(out!, a, b)
    @test out!.x == 0
    @test out!.gradient == [1, 3.0, 2.0]
    @test out!.hessian == [0 0 0; 0 0 1; 0 1 0.0]
end

@testset "hessian +" begin
    hdata = zeros(2,2)
    gdata = [1.0, 0]
    out! = HessianData(3.0, gdata, hdata, 1)
    a = HessianData(2.0, gdata, hdata, 2)
    @instr ⊖(identity)(out!, a)
    @test out!.x == 1
    @test out!.gradient == [1, 1.0]
    @test out!.hessian == [0 0; 0 0]
end

@testset "hessian ROT" begin
    hdata = zeros(3,3)
    gdata = [1.0, 0, 0]
    a! = 0.8
    b! = 1.5
    θ = 1.0
    res = taylor_hessian(ROT, (Loss(a!), b!, θ))
    @test res ≈ nhessian(ROT, (Loss(a!),b!,θ))
end

@testset "hessian" begin
    h1 = taylor_hessian(⊕(*), (Loss(0.0), 2.0, 3.0))
    h2 = nhessian(⊕(*), (Loss(0.0), 2.0, 3.0))
    @test h1 ≈ h2

    @i function test(a,b,c,d)
        a += b*c
        a += b^d
        ROT(a, c, d)
    end
    h1 = taylor_hessian(test, (Loss(0.0), 2.0, 1.0, 3.0))
    h2 = nhessian(test, (Loss(0.0), 2.0, 1.0, 3.0))
    @show h2
    @test h1 ≈ h2
end

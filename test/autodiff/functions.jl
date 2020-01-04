using NiLang, NiLang.AD
using Test, LinearAlgebra

@testset "inorm2, dot" begin
    out = 0.0
    vec = [1.0, 2.0, 3.0]
    vec2 = [1.0, 2.0, 5.0]
    @instr inorm2(out, vec)
    @test out ≈ norm(vec)^2
    @test check_inv(inorm2, (out, vec))

    out = 0.0
    vec = [1.0, 2.0, 3.0]
    vec2 = [1.0, 2.0, 5.0]
    @test check_grad(inorm2, (Loss(out), vec); verbose=true)

    out = 0.0
    @instr idot(out, vec, vec2)
    @test out ≈ dot(vec, vec2)
    @test check_inv(idot, (out, vec, vec2))

    @test check_grad(idot, (Loss(0.0), vec, vec2); verbose=true)

    m = randn(4,4)
    n = randn(4,4)
    out = 0.0
    @instr idot(out, m[:,2], n[:,4])
    @test out ≈ dot(m[:,2], n[:,4])
    @test check_inv(idot, (out, m[:,2], n[:,4]))

    @test check_grad(idot, (Loss(0.0), vec, vec2); verbose=true)
end

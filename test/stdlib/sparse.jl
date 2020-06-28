using NiLang
using NiSparseArrays
using SparseArrays
using Test, Random

@testset "dot" begin
    Random.seed!(2)
    sp1 = sprand(10, 10,0.3)
    sp2 = sprand(10, 10,0.3)
    @test SparseArrays.dot(sp1, sp2) ≈ NiSparseArrays.dot(0.0, sp1, sp2)[1]
end

@testset "mul!" begin
    Random.seed!(2)
    sp1 = sprand(10, 10,0.3)
    v = randn(10)
    out = zero(v)
    @test SparseArrays.mul!(copy(out), sp1, v, 0.5, 1) ≈ NiSparseArrays.mul!(copy(out), sp1, v, 0.5, 1)[1]
end


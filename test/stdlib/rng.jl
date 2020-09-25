using Test, NiLang, Random

@testset "LcgRNG" begin
    rng = LcgRNG(; seed=42)
    @test rng isa LcgRNG{UInt64}
    @test rng.x == 42
    @instr rand(rng)
    @test rng.x != 42
    @test check_inv(rand, (rng,))
end
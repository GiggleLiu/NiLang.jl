using Test, NiLang

@testset "@zeros" begin
    @test (@macroexpand @zeros Float64 a b c) == :(begin
        a ← zero(Float64)
        b ← zero(Float64)
        c ← zero(Float64)
    end)
end

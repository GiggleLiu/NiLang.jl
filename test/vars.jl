using Test, NiLang, NiLangCore

@testset "@zeros" begin
    @test (@macroexpand @zeros Float64 a b c) == :(begin
        a ← zero(Float64)
        b ← zero(Float64)
        c ← zero(Float64)
    end) |> NiLangCore.rmlines

    @test (@macroexpand @ones Float64 a b c) == :(begin
        a ← one(Float64)
        b ← one(Float64)
        c ← one(Float64)
    end) |> NiLangCore.rmlines
end

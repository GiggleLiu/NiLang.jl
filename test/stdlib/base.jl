using NiLang, NiLang.AD
using Test

@testset "sqdistance" begin
    @test i_sqdistance(0.0, [1.0, 0.0], [0.0, 1.0])[1] == 2.0
end

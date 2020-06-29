using NiLang, Test

@testset "filter and mapfoldl" begin
    @i function f(z, y, x)
        i_filter!((@skip! x -> x < 0), y, x)
        i_mapfoldl((@skip! exp), (@skip! PlusEq(identity)), z, y)
    end
    @test f(0.0, Float64[], [-1, -0.5, 3])[1] == exp(-0.5) + exp(-1)
end

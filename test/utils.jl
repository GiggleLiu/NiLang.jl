using NiLang
using Test

@testset "vec dataview" begin
    @i function f(x::AbstractVector, y::AbstractMatrix)
        x .+= (y |> vec)
        vec(y)[5] += x[4]
    end
    x = zeros(25)
    y = ones(5,5)
    z = ones(5,5)
    z[5] = 2.0
    @instr f(x, y)
    @test y == z
end

using NiLang
using Test

@testset "vec dataview" begin
    @i function f(x::AbstractVector, y::AbstractMatrix)
        x .+= identity.(vec(y))
        vec(y)[5] += identity(x[4])
    end
    x = zeros(25)
    y = ones(5,5)
    z = ones(5,5)
    z[5] = 2.0
    @instr f(x, y)
    @test y == z
end

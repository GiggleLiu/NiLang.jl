using NiLang
using Test

@testset "identity" begin
    x, y = 0.2, 0.5
    @instr x += identity(y)
    @test x == 0.7 && y==0.5
end

@testset "*, /" begin
    x, y, out = 2.0, 2.0, 1.0
    @instr out += x * y
    @test x == 2.0 && y == 2.0 && out == 5.0
    x, y, out = 2.0, 2.0, 1.0
    @instr out += x / y
    @test x == 2.0 && y == 2.0 && out == 2.0

    out = Fixed43(0.0)
    x = 1
    @instr out += x/2
    @test out === Fixed43(0.5)
    @instr out -= x/2
    @test out === Fixed43(0.0)
end

@testset "SWAP" begin
    x, y = 1, 2
    @instr SWAP(x, y)
    @test x == 2 && y == 1
end

@testset "NEG" begin
    x = 0.3
    @instr NEG(x)
    @test x == -0.3
    @test check_inv(NEG, (x,))
end

@testset "INV" begin
    x = 0.2
    @instr INV(x)
    @test x == 5.0
    @test check_inv(INV, (x,))
end

@testset "AddConst" begin
    x = 0.3
    @instr AddConst(4.0)(x)
    @test x == 4.3
    @test check_inv(AddConst(4.0), (x,))

    x = 0.3
    @instr SubConst(4.0)(x)
    @test x == -3.7
    @test check_inv(SubConst(4.0), (x,))
end

@testset "FLIP" begin
    x = false
    @instr FLIP(x)
    @test x == true
    @test check_inv(FLIP, (x,))
end

@testset "ROT" begin
    x, y, θ = 0.0, 1.0, π
    @test check_inv(ROT, (x, y, θ); verbose=true)
    @test check_inv(IROT, (x, y, θ); verbose=true)
end

@testset "INC, DEC" begin
    x = Int32(2)
    @instr INC(x)
    @test x === Int32(3)
    @instr DEC(x)
    @test x === Int32(2)
end

@testset "HADAMARD" begin
    x = 0.5
    y = 0.8
    @test check_inv(HADAMARD, (x, y))
end

@testset "dataviews" begin
    @i function f(z, y, x)
        y += cos(x |> INV)
        z += tan(y |> AddConst(4.0))
        z += y * (x |> NEG |> SubConst(0.5) |> INV)
        z += sin(x |> INV)
    end
    @test check_inv(f, (0.2, 0.5, 0.8))
end

@testset "fixed point arithmetics" begin
    for op in [exp, log, sin, sinh, asin, cos, cosh, acos, tan, tanh, atan]
        x, y = Fixed43(2.0), Fixed43(0.5)
        @instr x += op(y)
        @test x ≈ 2.0 + op(0.5)
    end
    for op in [SWAP, HADAMARD]
        x, y = Fixed43(2.0), Fixed43(0.5)
        @instr op(x, y)
        @test x ≈ op(2.0, 0.5)[1]
        @test y ≈ op(2.0, 0.5)[2]
    end
    for op in [NEG, INC, DEC]
        x = Fixed43(2.0)
        @instr op(x)
        @test x ≈ op(2.0)
    end
    for op in [^, /]
        x, y, z = Fixed43(2.0), Fixed43(0.5), Fixed43(0.8)
        @instr PlusEq(op)(x, y, z)
        @test x ≈ 2.0 + op(0.5, 0.8)
    end
end

@testset "additive identity" begin
    struct TestAdd{T}
        x::T
        y::Vector{T}
    end
    @test getfield.(PlusEq(identity)(TestAdd(1, [2]), TestAdd(10, [2])), :x) == (TestAdd(11, [4]).x, TestAdd(10, [2]).x)
    @test getfield.(PlusEq(identity)(TestAdd(1, [2]), TestAdd(10, [2])), :y) == (TestAdd(11, [4]).y, TestAdd(10, [2]).y)
end
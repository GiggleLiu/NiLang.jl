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

@testset "-" begin
    x = 0.3
    @instr -(x)
    @test x == -0.3
    @test check_inv(-, (x,))
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
    @instr (x,) |> INC |> DEC |> DEC
    @test x === Int32(1)
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
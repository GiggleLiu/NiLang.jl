using NiLang
using Test

@testset "auto bcast" begin
    a = AutoBcast([1.0, 2.0, 3.0])
    @instr NEG(a)
    @test a.x == [-1.0,-2.0,-3.0]
    a = AutoBcast([1.0, 2.0, 3.0])
    @instr INC(a)
    @test a.x == [2.0,3.0,4.0]
    @instr DEC(a)
    @test a.x == [1.0,2.0,3.0]

    a = AutoBcast([false, true, true])
    @instr FLIP(a)
    @test a.x == [true, false, false]

    a = AutoBcast([1.0, 2.0, 3.0])
    b = AutoBcast([1.0, 2.0, 4.0])
    @instr a += identity(b)
    @test a.x == [2,4,7.0]
    @test b.x == [1,2,4.0]
    @instr SWAP(a, b)
    @test b.x == [2,4,7.0]
    @test a.x == [1,2,4.0]

    a = AutoBcast([1.0, 2.0, 3.0])
    b = 2.0
    @instr a += identity(b)
    @test a.x == [3,4,5.0]
    @test b == 2.0

    a = AutoBcast([1.0, 2.0, 3.0])
    b = AutoBcast([1.0, 2.0, 4.0])
    c = AutoBcast([1.0, 2.0, 1.0])
    @instr a += b * c
    @test a.x == [2,6,7.0]
    @test b.x == [1,2,4.0]
    @test c.x == [1,2,1.0]

    a = AutoBcast([1.0, 2.0, 3.0])
    b = 2.0
    c = AutoBcast([1.0, 2.0, 1.0])
    @instr a += b * c
    @test a.x == [3,6,5.0]
    @test b == 2.0
    @test c.x == [1,2,1.0]

    a = AutoBcast([1.0, 2.0, 3.0])
    b = AutoBcast([1.0, 2.0, 4.0])
    c = 3.0
    @instr a += b * c
    @test a.x == [4,8,15.0]
    @test b.x == [1,2,4.0]
    @test c == 3.0

    a = AutoBcast([1.0, 2.0, 3.0])
    b = 2.0
    c = 3.0
    @instr a += b * c
    @test a.x == [7,8,9.0]
    @test b == 2.0
    @test c == 3.0

    @test zero(AutoBcast{Int,3}) == AutoBcast([0, 0, 0])
end

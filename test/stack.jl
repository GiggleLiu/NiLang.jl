using NiLang, Test
using NiLang: UNSAFE_POP!, UNSAFE_PUSH!, UNSAFE_COPYPOP!, UNSAFE_COPYPUSH!

@testset "stack operations" begin
    x =0.3
    @instr PUSH!(x)
    @test x === 0.0
    @instr POP!(x)
    @test x === 0.3
    @instr PUSH!(x)
    x = 0.4
    @test_throws InvertibilityError @instr POP!(x)
    y = 0.5
    @instr PUSH!(y)
    @instr UNSAFE_POP!(x)
    @test x == 0.5

    x =0.3
    st = Float64[]
    @instr PUSH!(st, x)
    @test x === 0.0
    @test length(st) == 1
    @instr POP!(st, x)
    @test length(st) == 0
    @test x === 0.3
    @instr PUSH!(st, x)
    @test length(st) == 1
    x = 0.4
    @test_throws InvertibilityError @instr POP!(x)
    @test length(st) == 1

    y = 0.5
    @instr PUSH!(st, y)
    @instr UNSAFE_POP!(st, x)
    @test x == 0.5

    @i function test(x)
        x2 ← zero(x)
        x2 += x^2
        PUSH!(x)
        SWAP(x, x2)
    end
    @test test(3.0) == 9.0
    l = length(NiLang.GLOBAL_STACK)
    @test check_inv(test, (3.0,))
    @test length(NiLang.GLOBAL_STACK) == l
end

@testset "copied push/pop stack operations" begin
    x =0.3
    @instr COPYPUSH!(x)
    @test x === 0.3
    @instr COPYPOP!(x)
    @test x === 0.3
    @instr COPYPUSH!(x)
    x = 0.4
    @test_throws InvertibilityError @instr COPYPOP!(x)
    y = 0.5
    @instr COPYPUSH!(y)
    @instr UNSAFE_COPYPOP!(x)
    @test x == 0.5

    x =0.3
    st = Float64[]
    @instr COPYPUSH!(st, x)
    @test x === 0.3
    @test length(st) == 1
    @instr COPYPOP!(st, x)
    @test length(st) == 0
    @test x === 0.3
    @instr COPYPUSH!(st, x)
    @test length(st) == 1
    x = 0.4
    @test_throws InvertibilityError @instr COPYPOP!(x)
    @test length(st) == 1

    y = 0.5
    @instr COPYPUSH!(st, y)
    @instr UNSAFE_COPYPOP!(st, x)
    @test x == 0.5

    @i function test(x, x2)
        x2 += x^2
        COPYPUSH!(x)
        SWAP(x, x2)
    end
    @test test(3.0, 0.0) == (9.0, 3.0)
    l = length(NiLang.GLOBAL_STACK)
    @test check_inv(test, (3.0, 0.0))
    @test length(NiLang.GLOBAL_STACK) == l
end

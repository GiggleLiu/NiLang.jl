using Test, NiLang

@testset "complex" begin
    a = 1.0+ 2im
    @instr (a |> real) += 2
    @instr (a |> imag) += 2
    @test a == 3.0 + 4im

    a = 1.0+ 2im
    @instr a += complex(2.0, 2.0)
    @test a == 3.0 + 4.0im
    @i function f(loss, a::Complex{T}, b) where T
        @routine begin
            c ← zero(a)
            sq ← zero(T)
            c += a * b
            sq += (c |> real) ^ 2
            sq += (c |> imag) ^ 2
        end
        loss += sq ^ 0.5
        ~@routine
    end
    a = 1.0 + 2.0im
    b = 2.0 + 1.0im
    loss = 0.0
    @instr f(loss, a, b)
    @test loss ≈ abs(a*b)
end

@testset "complex arithmetics" begin
    for op in [exp, log, identity]
        x, y = 2.0+1.0im, 0.5+0.2im
        @instr x += op(y)
        @test x ≈ 2.0+1.0im + op(0.5+0.2im)
    end
    for op in [SWAP, HADAMARD]
        x, y = 2.0+1.0im, 0.5+0.2im
        @instr op(x, y)
        @test x ≈ op(2.0+1.0im, 0.5+0.2im)[1]
        @test y ≈ op(2.0+1.0im, 0.5+0.2im)[2]
    end
    for op in [NEG, INC, DEC]
        x = 2.0+1.0im
        @instr op(x)
        @test x ≈ op(2.0+1.0im)
    end
    for op in [^, /, +, -]
        x, y, z = 2.0+1.0im, 0.5+0.2im, 0.8-2.0im
        @instr PlusEq(op)(x, y, z)
        @test x ≈ 2.0+1.0im + op(0.5+0.2im, 0.8-2.0im)
    end
end


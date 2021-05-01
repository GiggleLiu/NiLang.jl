using NiLang, NiLang.AD
using Test
using NiLang.AD: wrap_bcastgrad

@i function asarrayfunc(params; f, kwargs...)
    if (length(params) == 1, ~)
        f(params[1]; kwargs...)
    elseif (length(params) == 2, ~)
        f(params[1], params[2]; kwargs...)
    elseif (length(params) == 3, ~)
        f(params[1], params[2], params[3]; kwargs...)
    end
end

@testset "bcastgrad" begin
    T = AutoBcast{Int, 4}
    @test wrap_bcastgrad(T, ones(10)) == [GVar(1.0, AutoBcast(ones(4))) for i=1:10]
    @test wrap_bcastgrad(T, 3) == 3
    @test wrap_bcastgrad(T, NoGrad(3.0)) == 3.0
    @test wrap_bcastgrad(T, 3.0) == GVar(3.0, AutoBcast(ones(4)))
    @test wrap_bcastgrad(T, (3.0,)) == (GVar(3.0, AutoBcast(ones(4))),)
    @test wrap_bcastgrad(T, exp) == exp
    @test wrap_bcastgrad(T, Inv(exp)) == Inv(exp)
end

@testset "jacobians" begin
    for op in [PlusEq(*), PlusEq(/), PlusEq(^), ROT]
        j1 = jacobian(asarrayfunc, [0.3, 0.4, 2.0]; iin=1, f=op)
        j2 = NiLang.AD.jacobian_repeat(asarrayfunc, [0.3, 0.4, 2.0]; iin=1, f=op)
        @test j1 ≈ j2
    end

    for op in [PlusEq(identity), PlusEq(abs), SWAP, PlusEq(exp), PlusEq(log), PlusEq(sin), PlusEq(cos)]
        j1 = jacobian(asarrayfunc, [0.3, 0.4]; iin=1, f=op)
        j2 = NiLang.AD.jacobian_repeat(asarrayfunc, [0.3, 0.4]; iin=1, f=op)
        @test j1 ≈ j2
    end

    for op in [-]
        j1 = jacobian(asarrayfunc, [0.3]; iin=1, f=op)
        j2 = NiLang.AD.jacobian_repeat(asarrayfunc, [0.3]; iin=1, f=op)
        @test j1 ≈ j2
    end
end

@testset "nograd" begin
    @test AddConst(3.0)(NoGrad(2.0)) == NoGrad(5.0)
    @test SWAP(NoGrad(2.0), NoGrad(3.0)) == (NoGrad(3.0), NoGrad(2.0))
    @test PlusEq(*)(NoGrad(2.0), NoGrad(3.0), NoGrad(4.0)) == (NoGrad(14.0), NoGrad(3.0), NoGrad(4.0))
end

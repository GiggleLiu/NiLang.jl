using NiLang, NiLang.AD
using Test

@i function asarrayfunc(params; f, kwargs...)
    if (length(params) == 1, ~)
        f(params[1]; kwargs...)
    elseif (length(params) == 2, ~)
        f(params[1], params[2]; kwargs...)
    elseif (length(params) == 3, ~)
        f(params[1], params[2], params[3]; kwargs...)
    end
end

@testset "jacobians" begin
    for op in [⊕(*), ⊕(/), ⊕(^), ROT]
        j1 = jacobian(asarrayfunc, [0.3, 0.4, 2.0]; iin=1, f=op)
        j2 = NiLang.AD.jacobian_repeat(asarrayfunc, [0.3, 0.4, 2.0]; iin=1, f=op)
        @test j1 ≈ j2
    end

    for op in [⊕(identity), ⊕(abs), SWAP, ⊕(exp), ⊕(log), ⊕(sin), ⊕(cos)]
        j1 = jacobian(asarrayfunc, [0.3, 0.4]; iin=1, f=op)
        j2 = NiLang.AD.jacobian_repeat(asarrayfunc, [0.3, 0.4]; iin=1, f=op)
        @test j1 ≈ j2
    end

    for op in [NEG]
        j1 = jacobian(asarrayfunc, [0.3]; iin=1, f=op)
        j2 = NiLang.AD.jacobian_repeat(asarrayfunc, [0.3]; iin=1, f=op)
        @test j1 ≈ j2
    end
end

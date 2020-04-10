using NiLang, NiLang.AD
using Test

@testset "hessian propagate" begin
    for op in [⊕(*), ⊕(/), ⊕(^), ROT]
        j1 = jacobian(op, (0.3, 0.4, 2.0))
        j2 = NiLang.AD.jacobian_repeat(op, (0.3, 0.4, 2.0))
        @test j1 ≈ j2
    end

    for op in [⊕(identity), ⊕(abs), SWAP, ⊕(exp), ⊕(log), ⊕(sin), ⊕(cos)]
        j1 = jacobian(op, (0.3, 0.4))
        j2 = NiLang.AD.jacobian_repeat(op, (0.3, 0.4))
        @test j1 ≈ j2
    end

    for op in [NEG]
        j1 = jacobian(op, (0.3,))
        j2 = NiLang.AD.jacobian_repeat(op, (0.3,))
        @test j1 ≈ j2
    end
end

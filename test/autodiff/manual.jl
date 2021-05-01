using NiLang, Test
using NiLang.AD

test_func(x) = exp(x)
NiLang.AD.primitive_grad(::typeof(test_func), x) = exp(x)

test_g(x, y; k=0) = x^k * y
function NiLang.AD.primitive_grad(::typeof(test_g), x, y; k=0)
    return k*x^(k-1)*y, x^k
end

@testset "primitive grad" begin
    @test check_grad(PlusEq(test_func), (1.0, 1.0), iloss=1)
    @test check_grad(PlusEq(test_g), (1.0, 3.0, 2.0), k=2, iloss=1)
end

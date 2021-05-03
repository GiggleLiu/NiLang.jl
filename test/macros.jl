using Test, NiLang
using NiLang: auto_alloc, auto_expand

NiLang.alloc(::typeof(NiLang.i_sum), x::AbstractArray{T}) where T = zero(T)

@testset begin
    @test auto_alloc(:(y = exp(x))) == Expr(:block, :(y ← $alloc(exp, x)), :(y += exp(x)))
    ex1 = :(PlusEq(sin)(z, sin(x + 2y)))
    ex2 = auto_expand(ex1)
    @test length(ex2.args) == 13
    @i function test(x, y, z)
        #@auto_expand z += sin(x + 2y)
        @invcheckoff @auto_expand z += sin(x + 2y)
    end
    x, y, z = 1.0, 2.0, 3.0
    @test test(x, y, z)[3] == z + sin(x + 2y)
    @test check_inv(test, (x, y, z))
    @i function test(x, y, z, a)
        @auto_expand PlusEq(sin)(Complex{}(x, y), Complex{}(z, sin(a)))
    end
    x, y, z, a = 1.0, 2.0, 3.0, 4.0
    @test Complex(test(x, y, z, a)[1:2]...) == 1+im*y + sin(z+im*sin(a))
    @test check_inv(test, (x, y, z, a))

    @i function test2(y, x)
        @routine begin
            @auto_alloc i_sum(z, x)
        end
        y += z
        ~@routine
    end
    @test test2(1.0, [2,3.0])[1] == 6.0
    @test check_inv(test2, (1.0, [2,3.0]))
end
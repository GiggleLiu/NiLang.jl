using MatchCore, NiLang

function alloc end

macro auto_alloc(ex)
    esc(auto_alloc(ex))
end

function auto_alloc(ex)
    @smatch ex begin
        :($f($out, $(args...))) => begin
            Expr(:block, :($out ← $alloc($f, $(args...))), ex)
        end
        :($out = $f($(args...))) => begin
            if length(args) == 0
                error("number of arguments must be >= 1.")
            else
                Expr(:block, :($out ← $alloc($f, $(args...))), :($out += $f($(args...))))
            end
        end
        _ => error("can not allocate automatically for expression: `$ex`")
    end
end

for OPM in [:PlusEq, :MinusEq]
    for OP in [:+, :-, :*, :/, :^]
        @eval alloc(::$OPM{typeof($OP)}, x::T1, ::T2) where {T1<:Number,T2<:Number} = zero(promote_type(T1, T2))
    end
    for OP in [:sin, :cos, :tan, :asin, :atan, :acos, :sinh, :cosh, :tanh, :identity, :sqrt, :exp, :log]
        @eval alloc(::$OPM{typeof($OP)}, x::T) where T<:Number = zero(T)
    end
    for OP in [:abs, :abs2]
        @eval alloc(::$OPM{typeof($OP)}, x::T) where T<:Number = zero(real(T))
    end
    @eval alloc(::$OPM{typeof(sincos)}, x::T) where T<:Number = (zero(T), zero(T))
end

function auto_expand(ex)
    res = Expr[]
    auto_expand!(copy(ex), res)
    Expr(:block, res..., NiLangCore.dual_body(@__MODULE__, res[1:end-1])...)
end

function auto_expand!(ex, exprs, sym=nothing, addnew=true)
    @smatch ex begin
        :($f($(args...))) => begin
            for (i, arg) in enumerate(args)
                @smatch arg begin
                    :($_{$(_...)}($(_...))) => begin
                        auto_expand!(arg, exprs, nothing, false)
                    end
                    :($f2($(vs...))) => begin
                        sym2 = gensym()
                        auto_expand!(:(PlusEq($f2)($sym2, $(vs...))), exprs, sym2, true)
                        args[i] = sym2
                    end
                    _ => nothing
                end
            end
            if sym !== nothing
                push!(exprs, :($sym ← $alloc($f, $(args[2:end]...))))
            end
            if addnew
                push!(exprs, :($f($(args...))))
            end
        end
        :($a += $b) || :($a -= $b) || :($a *= $b) || :($a /= $b) || :($a ⊻= $b) => begin
            auto_expand!(NiLangCore.to_standard_format(ex), exprs, sym, addnew)
        end
        _ => error("Can only expand an expression like `f(args...)`, got $(ex)!")
    end
end

macro auto_expand(ex)
    esc(auto_expand(ex))
end

using Test

alloc(::typeof(NiLang.i_sum), x::AbstractArray{T}) where T = zero(T)

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
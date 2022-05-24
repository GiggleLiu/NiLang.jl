using MLStyle, NiLang
export alloc, @auto_alloc, @auto_expand

"""
    alloc(f, args...)

allocate function output space (the first argument), where `args` only contains the last `N-1` arguments.
"""
function alloc end

macro auto_alloc(ex)
    esc(auto_alloc(ex))
end

function auto_alloc(ex)
    @match ex begin
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
    @match ex begin
        :($f($(args...))) => begin
            for (i, arg) in enumerate(args)
                @match arg begin
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

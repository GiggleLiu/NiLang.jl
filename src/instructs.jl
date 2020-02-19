export XOR, SWAP, NEG, CONJ
export ROT, IROT
export ipop!, ipush!

const GLOBAL_STACK = []

############# global stack operations ##########
@inline function ipush!(x)
    push!(GLOBAL_STACK, x)
    zero(x)
end
@inline function ipop!(x)
    @invcheck x zero(x)
    pop!(GLOBAL_STACK)
end

############# local stack operations ##########
@inline function ipush!(stack, x)
    push!(stack, x)
    stack, zero(x)
end
@inline function ipop!(stack, x)
    @invcheck x zero(x)
    stack, pop!(stack)
end
@dual ipop! ipush!

"""
    NEG(a!) -> -a!
"""
@inline function NEG(a!::Number)
    -a!
end
@selfdual NEG

"""
    CONJ(a!) -> a!'
"""
@inline function CONJ(a!::Number)
    conj(a!)
end
@selfdual CONJ

"""
    XOR(a!, b) -> a! ⊻ b, b
"""
@inline function XOR(a!::Number, b::Number)
    a! ⊻ b, b
end
@selfdual XOR

"""
    SWAP(a!, b!) -> b!, a!
"""
@inline function SWAP(a!::Number, b!::Number)
    b!, a!
end
@selfdual SWAP

"""
    ROT(a!, b!, θ) -> a!', b!', θ

```math
\\begin{align}
    {\\rm ROT}(a!, b!, \\theta)  = \\begin{bmatrix}
        \\cos(\\theta) & - \\sin(\\theta)\\\\
        \\sin(\\theta)  & \\cos(\\theta)
    \\end{bmatrix}
    \\begin{bmatrix}
        a!\\\\
        b!
    \\end{bmatrix},
\\end{align}
```
"""
@inline function ROT(i::Number, j::Number, θ::Number)
    a, b = rot(i, j, θ)
    a, b, θ
end

"""
    IROT(a!, b!, θ) -> ROT(a!, b!, -θ)
"""
@inline function IROT(i::Number, j::Number, θ::Number)
    i, j, _ = ROT(i, j, -θ)
    i, j, θ
end
@dual ROT IROT

for F1 in [:NEG, :CONJ]
    @eval @i @inline function $F1(a!)
        $F1(value(a!))
    end
    @eval NiLangCore.nouts(::typeof($F1)) = 1
    @eval NiLangCore.nargs(::typeof($F1)) = 1
end

for F2 in [:XOR, :SWAP]
    @eval @i @inline function $F2(a, b)
        $F2(value(a), value(b))
    end
    @eval NiLangCore.nouts(::typeof($F2)) = $(F2 == :SWAP ? 2 : 1)
    @eval NiLangCore.nargs(::typeof($F2)) = 2
end

for F3 in [:ROT, :IROT]
    @eval @inline @generated function $F3(a!, b!, θ)
        if !(a! <: IWrapper || b! <: IWrapper || θ <: IWrapper)
            return :(MethodError($($F3), (a!, b!, θ)))
        end
        param_a = a! <: IWrapper ? :(value(a!)) : :(a!)
        param_b = b! <: IWrapper ? :(value(b!)) : :(b!)
        param_θ = θ <: IWrapper ? :(value(θ)) : :(θ)
        quote
            @instr $($F3)($param_a, $param_b, $param_θ)
            a!, b!, θ
        end
    end
    @eval NiLangCore.nouts(::typeof($F3)) = 2
    @eval NiLangCore.nargs(::typeof($F3)) = 3
end

for (TP, OP) in [(:PlusEq, :+), (:MinusEq, :-), (:XorEq, :⊻)]
    @eval @inline @generated function (inf::$TP)(out!, x; kwargs...)
        if !(out! <: IWrapper || x <:IWrapper)
            return :(MethodError(inf, (out!, x)))
        end
        param_out = out! <: IWrapper ? :(value(out!)) : :(out!)
        param_x = x <: IWrapper ? :(value(x)) : :(x)
        quote
            @instr inf($param_out, $param_x, kwargs...)
            out!, x
        end
    end
    @eval @inline @generated function (inf::$TP)(out!, x, y; kwargs...)
        if !(out! <: IWrapper || x <:IWrapper || y<:IWrapper)
            return :(MethodError(inf, (out!, x, y)))
        end
        param_out = out! <: IWrapper ? :(value(out!)) : :(out!)
        param_x = x <: IWrapper ? :(value(x)) : :(x)
        param_y = x <: IWrapper ? :(value(y)) : :(y)
        quote
            @instr inf($param_out, $param_x, $param_y; kwargs...)
            out!, x, y
        end
    end
    @eval NiLangCore.nouts(::$TP) = 1

    for SOP in [:*, :/, :^]
        @eval NiLangCore.nargs(::$TP{typeof($SOP)}) = 3
    end
    for SOP in [:sin, :cos, :log, :exp, :identity, :abs]
        @eval NiLangCore.nargs(::$TP{typeof($SOP)}) = 2
    end
end

# patch for fixed point numbers
function (f::PlusEq{typeof(/)})(out!::T, x::Integer, y::Integer) where T<:Fixed
    out!+T(x)/y, x, y
end

function (f::MinusEq{typeof(/)})(out!::T, x::Integer, y::Integer) where T<:Fixed
    out!-T(x)/y, x, y
end

for F in [:exp, :log, :sin, :cos]
    @eval Base.$F(x::Fixed43) = Fixed43($F(Float64(Fixed43(x))))
end

Base.:^(x::Integer, y::Fixed43) = Fixed43(x^(Float64(y)))
Base.:^(x::Fixed43, y::Fixed43) = Fixed43(x^(Float64(y)))
Base.:^(x::T, y::Fixed43) where T<:AbstractFloat = x^(T(y))

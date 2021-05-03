export SWAP, FLIP
export ROT, IROT
export INC, DEC, NEG, INV, AddConst, SubConst
export HADAMARD

"""
    NoGrad{T} <: IWrapper{T}
    NoGrad(x)

A `NoGrad(x)` is equivalent to `GVar^{-1}(x)`, which cancels the `GVar` wrapper.
"""
struct NoGrad{T} <: IWrapper{T}
    x::T
end
NoGrad(x::NoGrad{T}) where T = x # to avoid ambiguity error
NoGrad{T}(x::NoGrad{T}) where T = x # to avoid ambiguity error
(_::Type{Inv{NoGrad}})(x) = x.x
@fieldview value(x::NoGrad) = x.x

const NullType{T} = Union{NoGrad{T}, Partial{T}}

NEG(a!) = -(a!)
@selfdual NEG
@selfdual -

INV(a!) = inv(a!)
@selfdual INV

@inline FLIP(b::Bool) = !b
@selfdual FLIP

"""
    INC(a!) -> a! + 1
"""
@inline function INC(a!::Number)
    a! + one(a!)
end

"""
    DEC(a!) -> a! - 1
"""
@inline function DEC(a!::Number)
    a! - one(a!)
end
@dual INC DEC


"""
    SWAP(a!, b!) -> b!, a!
"""
@inline function SWAP(a!::T, b!::T) where T
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
@inline function ROT(i::Real, j::Real, θ::Real)
    a, b = rot(i, j, θ)
    a, b, θ
end

"""
    IROT(a!, b!, θ) -> ROT(a!, b!, -θ)
"""
@inline function IROT(i::Real, j::Real, θ::Real)
    i, j, _ = ROT(i, j, -θ)
    i, j, θ
end
@dual ROT IROT

"""
    HADAMARD(x::Real, y::Real)

Hadamard transformation that returns `(x + y)/√2, (x - y)/√2`
"""
function HADAMARD(x::Real, y::Real)
    sqrt(0.5) * (x + y), sqrt(0.5) * (x - y)
end

@selfdual HADAMARD

# more data views
for (DT, OP, NOP) in [(:AddConst, :+, :-), (:SubConst, :-, :+)]
    @eval struct $DT{T}
        x::T
    end

    @eval function (f::$DT)(y::Real)
        $OP(y, f.x)
    end

    @eval NiLangCore.chfield(x::T, ac::$DT, xval::T) where T<:Real = $NOP(xval, ac.x)
end

for F1 in [:(Base.:-), :NEG, :(ac::AddConst), :(sc::SubConst)]
    @eval @inline function $F1(a!::NullType)
        @instr $F1(a! |> value)
        a!
    end
end

for F2 in [:SWAP, :HADAMARD, :((inf::PlusEq)), :((inf::MinusEq)), :((inf::XorEq))]
    @eval @inline function $F2(a::NullType, b::Real)
        @instr $(NiLangCore.get_argname(F2))(a |> value, b)
        a, b
    end
    @eval @inline function $F2(a::NullType, b::NullType)
        @instr $(NiLangCore.get_argname(F2))(a |> value, b |> value)
        a, b
    end
    @eval @inline function $F2(a::Real, b::NullType)
        @instr $(NiLangCore.get_argname(F2))(a, b |> value)
        a, b
    end
end

function type_except(::Type{TT}, ::Type{T2}) where {TT, T2}
    N = length(TT.parameters)
    setdiff(Base.Iterators.product(zip(TT.parameters, repeat([T2], N))...), [ntuple(x->T2, N)])
end

for F3 in [:ROT, :IROT, :((inf::PlusEq)), :((inf::MinusEq)), :((inf::XorEq))]
    PS = (:a, :b, :c)
    for PTS in type_except(Tuple{NullType, NullType, NullType}, Real)
        params = map((P,PT)->PT <: NullType ? :($P |> value) : P, PS, PTS)
        params_ts = map((P,PT)->:($P::$PT), PS, PTS)
        @eval @inline function $F3($(params_ts...))
            @instr $F3($(params...))
            ($(PS...),)
        end
    end
end

# patch for fixed point numbers
function (f::PlusEq{typeof(/)})(out!::T, x::Integer, y::Integer) where T<:Fixed
    out!+T(x)/y, x, y
end

function (f::MinusEq{typeof(/)})(out!::T, x::Integer, y::Integer) where T<:Fixed
    out!-T(x)/y, x, y
end

for F in [:exp, :log, :sin, :sinh, :asin, :cos, :cosh, :acos, :tan, :tanh, :atan]
    @eval Base.$F(x::Fixed43) = Fixed43($F(Float64(x)))
    @eval (f::PlusEq{typeof($F)})(out!::Fixed43, x::Real) = out! + Fixed43($F(x)), x
    @eval (f::MinusEq{typeof($F)})(out!::Fixed43, x::Real) = out! - Fixed43($F(x)), x
end

Base.:^(x::Integer, y::Fixed43) = Fixed43(x^(Float64(y)))
Base.:^(x::Fixed43, y::Fixed43) = Fixed43(x^(Float64(y)))
Base.:^(x::T, y::Fixed43) where T<:AbstractFloat = x^(T(y))

function (::PlusEq{typeof(convert)})(out!::T, y) where T<:Real
    out! + convert(T, y), y
end

function (::MinusEq{typeof(convert)})(out!::T, y) where T<:Real
    out! - convert(T, y), y
end

Base.:~(ac::AddConst) = SubConst(ac.x)
Base.:~(ac::SubConst) = AddConst(ac.x)
@dualtype AddConst SubConst

for F in [:INV, :NEG, :FLIP, :INC, :DEC]
    @eval NiLangCore.chfield(x::T, ::typeof($F), xval::T) where T<:Real = (~$F)(xval)
end

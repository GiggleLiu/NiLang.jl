export SWAP, FLIP
export ROT, IROT
export INC, DEC

"""
    NoGrad{T} <: IWrapper{T}
    NoGrad(x)

A `NoGrad(x)` is equivalent to `GVar^{-1}(x)`, which cancels the `GVar` wrapper.
"""
@pure_wrapper NoGrad

const NullType{T} = Union{NoGrad{T}, Partial{T}}

@selfdual -

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

for F1 in [:(Base.:-)]
    @eval @inline function $F1(a!::NullType)
        @instr $F1(a! |> value)
        a!
    end
end

for F2 in [:SWAP, :((inf::PlusEq)), :((inf::MinusEq)), :((inf::XorEq))]
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

for F in [:exp, :log, :sin, :cos]
    @eval Base.$F(x::Fixed43) = Fixed43($F(Float64(Fixed43(x))))
end

Base.:^(x::Integer, y::Fixed43) = Fixed43(x^(Float64(y)))
Base.:^(x::Fixed43, y::Fixed43) = Fixed43(x^(Float64(y)))
Base.:^(x::T, y::Fixed43) where T<:AbstractFloat = x^(T(y))

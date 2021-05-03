export IWrapper, Partial, unwrap, value

"""
    value(x)

Get the `value` from a wrapper instance.
"""
value(x) = x
NiLangCore.chfield(x::T, ::typeof(value), y::T) where T = y

"""
    IWrapper{T} <: Real

IWrapper{T} is a wrapper of for data of type T.
It will forward `>, <, >=, <=, ≈` operations.
"""
abstract type IWrapper{T} <: Real end
NiLangCore.chfield(x, ::Type{T}, v) where {T<:IWrapper} = (~T)(v)
Base.eps(::Type{<:IWrapper{T}}) where T = Base.eps(T)

"""
    unwrap(x)

Unwrap a wrapper instance (recursively) to get the content value.
"""
unwrap(x::IWrapper) = unwrap(value(x))
unwrap(x) = x

for op in [:>, :<, :>=, :<=, :isless, :(==)]
    @eval Base.$op(a::IWrapper, b::IWrapper) = $op(unwrap(a), unwrap(b))
    @eval Base.$op(a::IWrapper, b::Real) = $op(unwrap(a), b)
    @eval Base.$op(a::IWrapper, b::AbstractFloat) = $op(unwrap(a), b)
    @eval Base.$op(a::Real, b::IWrapper) = $op(a, unwrap(b))
    @eval Base.$op(a::AbstractFloat, b::IWrapper) = $op(a, unwrap(b))
end

"""
Partial{FIELD, T, T2} <: IWrapper{T2}

Take a field `FIELD` without dropping information.
This operation can be undone by calling `~Partial{FIELD}`.
"""
struct Partial{FIELD, T, T2} <: IWrapper{T2}
    x::T
    function Partial{FIELD,T,T2}(x::T) where {T,T2,FIELD}
        new{FIELD,T,T2}(x)
    end
    function Partial{FIELD,T,T2}(x::T) where {T<:Complex,T2,FIELD}
        new{FIELD,T,T2}(x)
    end
end
Partial{FIELD}(x::T) where {T,FIELD} = Partial{FIELD,T,typeof(getfield(x,FIELD))}(x)
Partial{FIELD}(x::T) where {T<:Complex,FIELD} = Partial{FIELD,T,typeof(getfield(x,FIELD))}(x)

@generated function (_::Type{Inv{Partial{FIELD}}})(x::Partial{FIELD}) where {FIELD}
    :(x.x)
end

function NiLangCore.chfield(hd::Partial{FIELD}, ::typeof(value), val) where FIELD
    chfield(hd, Val(:x), chfield(hd.x, Val(FIELD), val))
end

@generated function value(hv::Partial{FIELD}) where FIELD
    :(hv.x.$FIELD)
end

function Base.zero(x::T) where T<:Partial
    zero(T)
end

function Base.zero(x::Type{<:Partial{FIELD,T}}) where {FIELD, T}
    Partial{FIELD}(Base.zero(T))
end
Base.show(io::IO, gv::Partial{FIELD}) where FIELD = print(io, "$(gv.x).$FIELD")
Base.show(io::IO, ::MIME"plain/text", gv::Partial) = Base.show(io, gv)
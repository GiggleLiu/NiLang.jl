######## GVar, a bundle that records gradient
"""
    GVar{T,GT} <: IWrapper{T}
    GVar(x)

Add gradient information to variable `x`, where `x` can be a real number or a general structure.
If it is a non-integer real number, it will wrap the element with a gradient field,
otherwise it will propagate into the type and wrap the elements with `GVar`.
Runing a program backward will update the gradient fields of `GVar`s. The following is a toy using case.

### Example

```jldoctest; setup=:(using NiLang)
julia> using NiLang.AD: GVar, grad

julia> struct A{T}
           x::T
       end

julia> GVar(A(2.0+3im), A(3.0+3im))
A{Complex{GVar{Float64, Float64}}}(GVar(2.0, 3.0) + GVar(3.0, 3.0)*im)

julia> @i function f(a::A, b::A)
           a.x += log(b.x)
       end

julia> outputs = f(A(2.0+3im), A(2.0-1im))  # forward pass
(A{ComplexF64}(2.8047189562170503 + 2.536352390999194im), A{ComplexF64}(2.0 - 1.0im))

julia> outputs_with_gradients = (GVar(outputs[1], A(3.0+3im)), GVar(outputs[2]))  # wrap `GVar`
(A{Complex{GVar{Float64, Float64}}}(GVar(2.8047189562170503, 3.0) + GVar(2.536352390999194, 3.0)*im), A{Complex{GVar{Float64, Float64}}}(GVar(2.0, 0.0) - GVar(1.0, -0.0)*im))

julia> inputs_with_gradients = (~f)(outputs_with_gradients...)  # backward pass
(A{Complex{GVar{Float64, Float64}}}(GVar(2.0, 3.0) + GVar(3.0, 3.0)*im), A{Complex{GVar{Float64, Float64}}}(GVar(2.0, 1.8) - GVar(1.0, -0.6000000000000002)*im))

julia> grad(inputs_with_gradients)
(A{ComplexF64}(3.0 + 3.0im), A{ComplexF64}(1.8 + 0.6000000000000002im))
```

The outputs of `~f` are gradients for input variables, one can use `grad` to take the gradient fields recursively.
"""
struct GVar{T,GT} <: IWrapper{T}
    x::T
    g::GT
    function GVar{T,GT}(x::T, g::GT) where {T,GT}
        new{T,GT}(x, g)
    end
    function GVar(x::T, g::T) where T<:Real
        new{T,T}(x, g)
    end
    function GVar{T,GT}(x::T2) where {T,T2,GT}
        new{T,GT}(T(x), zero(GT))
    end
    function GVar(x::T, g::GT) where {T,GT}
        new{T,GT}(x, g)
    end
end

# `GVar` and `~GVar` on composite types
@generated function GVar(x::Type{T}) where T
    :($(getfield(T.name.module, nameof(T))){$(GVar.(T.parameters)...)})
end
@generated function GVar(x::Type{T}, y::Type{T}) where T
    :($(getfield(T.name.module, nameof(T))){$(GVar.(T.parameters, T.parameters)...)})
end
@generated function (_::Type{Inv{GVar}})(x::Type{T}) where T
    :($(getfield(T.name.module, nameof(T))){$((~GVar).(T.parameters)...)})
end
# `GVar` and `~GVar` on composite vars
@generated function GVar(x::T) where T
    Expr(:new, GVar(T), [:(GVar(x.$NAME)) for NAME in fieldnames(T)]...)
end
@generated function GVar(x::T, g::T) where T
    Expr(:new, GVar(T, T), [:(GVar(x.$NAME, g.$NAME)) for NAME in fieldnames(T)]...)
end
@generated function (_::Type{Inv{GVar}})(x::T) where T
    Expr(:new, (~GVar)(T), [:((~GVar)(x.$NAME)) for NAME in fieldnames(T)]...)
end

for T in [:Real]
    ## differentiable elementary types
    @eval GVar(::Type{ET}) where ET<:$T = GVar{ET,ET}
    @eval GVar(::Type{ET}, ::Type{ET}) where ET<:$T = GVar{ET,ET}
    @eval (_::Type{Inv{GVar}})(::Type{GVar{ET,GT}}) where {ET<:$T,GT} = ET

    ## differentiable elementary vars
    @eval GVar(x::$T) = GVar(x, zero(x))
    @eval @inline function (_::Type{Inv{GVar}})(x::GVar{<:$T})
        @invcheck x.g zero(x.x)
        x.x
    end
end

for T in [:Integer, :Bool, :Function, :String, :Char, :Nothing]
    ## non-differentiable elementary types
    @eval GVar(::Type{ET}) where ET<:$T = ET
    @eval GVar(::Type{ET}, ::Type{ET}) where ET<:$T = GVar{ET,ET}
    @eval (_::Type{Inv{GVar}})(::Type{ET}) where ET<:$T = ET

    ## non-differentiable elementary vars
    @eval GVar(x::$T) = x
    @eval (_::Type{Inv{GVar}})(x::$T) = x
end

for T in [:Tuple, :AbstractArray]
    ## broadcastable elementary types
    @eval GVar(x::$T) = GVar.(x)
    @eval GVar(x::$T, y::$T) = GVar.(x, y)
    @eval (_::Type{Inv{GVar}})(x::$T) = (~GVar).(x)
end

# no gradient wrapper
GVar(x::NoGrad) = (~NoGrad)(x)

# define on complex numbers to fix ambiguity errors
GVar(x::Complex) = Complex(GVar(x.re), GVar(x.im))
GVar(x::Complex, y::Complex) = Complex(GVar(x.re, y.re), GVar(x.im, y.im))
(_::Type{Inv{GVar}})(x::Complex) = Complex((~GVar)(x.re), (~GVar)(x.im))

Base.copy(b::GVar) = GVar(b.x, copy(b.g))
Base.zero(x::GVar) = GVar(Base.zero(x.x), Base.zero(x.g))
Base.zero(::Type{<:GVar{T,GT}}) where {T,GT} = GVar(zero(T), zero(GT))
Base.one(x::GVar) = GVar(Base.one(x.x), Base.zero(x.g))
Base.one(::Type{<:GVar{T}}) where T = GVar(one(T))
Base.adjoint(b::GVar) = GVar(b.x', b.g')
Base.:-(b::GVar) = GVar(-b.x, -b.g)
Base.isapprox(x::GVar, y::GVar; kwargs...) = isapprox(x.x, y.x; kwargs...) && isapprox(x.g, y.g; kwargs...)

# define kernel and field views
"""
    grad(var)

Get the gradient field of `var`.
"""
@fieldview grad(gv::GVar) = gv.g
@fieldview value(gv::GVar) = gv.x
# TODO: fix the problem causing this patch, the field type can not change?!
chfield(x::GVar, ::typeof(value), xval::GVar) = GVar(xval, x.g)

@generated function grad(x::T) where T
    isprimitivetype(T) && throw("not supported type to obtain gradients: $T.")
    Expr(:new, typegrad(T), [:(grad(x.$NAME)) for NAME in fieldnames(T)]...)
end
typegrad(x) = x
@generated function typegrad(x::Type{T}) where T
    if isprimitivetype(T)
        T
    else
        :($(getfield(T.name.module, nameof(T))){$(typegrad.(T.parameters)...)})
    end
end
typegrad(::Type{GVar{ET,GT}}) where {ET,GT} = ET
grad(gv::T) where T<:Real = zero(T)
grad(gv::AbstractArray{T}) where T = grad.(gv)
grad(gv::Function) = 0
grad(gv::String) = ""
grad(t::Tuple) = grad.(t)
chfield(x::T, ::typeof(grad), g::T) where T = (@invcheck g zero(g); x)
chfield(x::GVar, ::typeof(grad), g::GVar) = GVar(x.x, g)
#chfield(x::GVar, ::typeof(-), val::GVar) = GVar(-val.x, -val.g)
chfield(x::Complex{<:GVar}, ::typeof(grad), g::Complex) = Complex(GVar(value(x.re), g.re), GVar(value(x.im), g.im))

# NOTE: superwarning: check value only to make ancilla gradient descardable.
NiLangCore.deanc(x::GVar{T}, val::GVar{T}) where T = NiLangCore.deanc(value(x), value(val))
function deanc(x::T, val::T) where {T<:AbstractArray}
   x === val || deanc.(x, val)
end

# constructors and deconstructors
Base.iszero(x::GVar) = iszero(x.x)

## variable mapping
function (_::Type{Inv{GVar}})(x::GVar{<:GVar,<:GVar})
    Partial{:x}(x)
end

Base.show(io::IO, gv::GVar) = print(io, "GVar($(gv.x), $(gv.g))")
Base.show(io::IO, ::MIME"plain/text", gv::GVar) = Base.show(io, gv)

# used in log number iszero function.
Base.isfinite(x::GVar) = isfinite(x.x)
# interfaces

_replace_opmx_callable(ex) = @smatch ex begin
    :(:+=($f)) => :(PlusEq($f))
    :(:-=($f)) => :(MinusEq($f))
    :(:*=($f)) => :(MulEq($f))
    :(:/=($f)) => :(DivEq($f))
    :(:⊻=($f)) => :(XorEq($f))
    _ => ex
end

"""
    @nograd f(args...)

Mark `f(args...)` as having no gradients.
"""
macro nograd(ex)
    @smatch ex begin
        :($f($(args...))) => begin
            f2 = _replace_opmx_callable(f)
            newargs = []
            for arg in args
                push!(newargs, @smatch arg begin
                    :($x::GVar) => :($x.x)
                    :($x::VecGVar) => :($x.x)
                    :($x::GVar{$tp}) => :($x.x)
                    _ => NiLangCore.get_argname(arg)
                end
                )
            end
            esc(quote
                @i function $f($(args...))
                    $f2($(newargs...))
                end
            end)
        end
        _ => error("expect `f(args...)`, got $ex")
    end
end

# ULogarithmic
_content(x::ULogarithmic) = x.log
NiLang.AD.GVar(x::ULogarithmic) = exp(ULogarithmic, GVar(_content(x), zero(_content(x))))
(_::Type{Inv{GVar}})(x::ULogarithmic{GVar{TE}}) where TE = exp(ULogarithmic{TE}, (~GVar)(_content(x)))

Base.one(x::ULogarithmic{GVar{T,GT}}) where {T, GT} = one(ULogarithmic{GVar{T,GT}})
Base.one(::Type{ULogarithmic{GVar{T,GT}}}) where {T,GT} = exp(ULogarithmic, GVar(zero(T), zero(GT)))
Base.zero(x::ULogarithmic{GVar{T,GT}}) where {T,GT} =zero(ULogarithmic{GVar{T,GT}})
Base.zero(::Type{ULogarithmic{GVar{T,T}}}) where T = exp(ULogarithmic, GVar(zero(T), zero(T)))

# the patch for dicts
function GVar(d::Dict)
    Dict([(k=>GVar(v)) for (k, v) in d])
end

function (_::Type{Inv{GVar}})(d::Dict)
    Dict([(k=>(~GVar)(v)) for (k, v) in d])
end

function grad(d::Dict)
    Dict([(k=>grad(v)) for (k, v) in d])
end

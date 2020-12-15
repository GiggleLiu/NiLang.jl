######## GVar, a bundle that records gradient
"""
    GVar{T,GT} <: IWrapper{T}
    GVar(x)

Attach a gradient field to `x`.
"""
@i struct GVar{T,GT} <: IWrapper{T}
    x::T
    g::GT
    function GVar{T,GT}(x::T, g::GT) where {T,GT}
        new{T,GT}(x, g)
    end
    function GVar{T,GT}(x::T) where {T, GT}
        new{T,GT}(x, zero(GT))
    end
    function GVar(x::T, g::GT) where {T,GT}
        new{T,GT}(x, g)
    end
    @i function GVar(x::Integer)
    end
    @i function GVar(x::Bool)
    end
    @i function GVar(x::Function)
    end
    @i function GVar(x::Tuple)
        GVar.(x)
    end
    @i function GVar(x::NoGrad)
        (~NoGrad)(x)
    end
end

function GVar(x::T) where T<:Real
    GVar(x, zero(x))
end

function (_::Type{Inv{GVar}})(x::GVar{T}) where T<:Real
    @invcheck x.g zero(x.x)
    x.x
end

@generated function GVar(x::T) where T
    quote
        $(getfield(T.name.module, nameof(T)))($([:(GVar(getfield(x, $(QuoteNode(NAME))))) for NAME in fieldnames(T)]...))
    end
end
@generated function (_::Type{Inv{GVar}})(x::T) where T
    quote
        $(getfield(T.name.module, nameof(T)))($([:((~GVar)(getfield(x, $(QuoteNode(NAME))))) for NAME in fieldnames(T)]...))
    end
end
GVar(x::Complex) = Complex(GVar(x.re), GVar(x.im))
GVar(x::Complex, y::Complex) = Complex(GVar(x.re, y.re), GVar(x.im, y.im))
(_::Type{Inv{GVar}})(x::Complex) = Complex((~GVar)(x.re), (~GVar)(x.im))
GVar(x::AbstractArray) = GVar.(x)
GVar(x::AbstractArray, y::AbstractArray) = GVar.(x, y)
(_::Type{Inv{GVar}})(x::AbstractArray) = (~GVar).(x)

Base.copy(b::GVar) = GVar(b.x, copy(b.g))
Base.zero(x::GVar) = GVar(Base.zero(x.x), Base.zero(x.g))
Base.zero(::Type{<:GVar{T,GT}}) where {T,GT} = GVar(zero(T), zero(GT))
Base.one(x::GVar) = GVar(Base.one(x.x), Base.zero(x.g))
Base.one(::Type{<:GVar{T}}) where T = GVar(one(T))
Base.adjoint(b::GVar) = GVar(b.x', b.g')
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
    quote
        $(getfield(T.name.module, nameof(T)))($([:(grad(getfield(x, $(QuoteNode(NAME))))) for NAME in fieldnames(T)]...))
    end
end
grad(gv::T) where T<:Real = zero(T)
grad(gv::AbstractArray{T}) where T = grad.(gv)
grad(gv::Function) = 0
grad(gv::String) = 0
chfield(x::T, ::typeof(grad), g::T) where T = (@invcheck iszero(g) || g≈0; x)
chfield(x::GVar, ::typeof(grad), g::GVar) where T = GVar(x.x, g)
chfield(x::Complex{<:GVar}, ::typeof(grad), g::Complex) where T = Complex(GVar(value(x.re), g.re), GVar(value(x.im), g.im))

# NOTE: superwarning: check value only to make ancilla gradient descardable.
NiLangCore.deanc(x::GVar, val::GVar) = NiLangCore.deanc(value(x), value(val))
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

"""
    @nograd f(args...)

Mark `f(args...)` as having no gradients.
"""
macro nograd(ex)
    @smatch ex begin
        :($f($(args...))) => begin
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
                    $f($(newargs...))
                end
            end)
        end
        _ => error("expect `f(args...)`, got $ex")
    end
end

# load data from stack
function loaddata(::Type{TG}, x::T) where {T,TG<:GVar{T}}
    TG(x)
end

function loaddata(::Type{T}, x::T) where T <: GVar
    x
end

function loaddata(::Type{AGT}, x::AT) where {T, GT, AT<:AbstractArray{T}, AGT<:AbstractArray{GVar{T,T}}}
    map(x->GVar(x, zero(x)), x)
end

#=
Base.convert(::Type{GVar{Tx, Tg}}, x::GVar) where {Tx, Tg} = GVar(convert(Tx, x.x), convert(Tg, x.g))
function Base.convert(::Type{GVar{Tx, Tg}}, x::ULogarithmic{<:GVar}) where {Tx, Tg}
    expx = exp(x.log.x)
    @show expx, Tx, Tg, x.log.g/expx
    @show GVar(convert(Tx, expx), convert(Tg, x.log.g/expx))
    GVar(convert(Tx, expx), convert(Tg, x.log.g/expx))
end
=#

# ULogarithmic
_content(x::ULogarithmic) = x.log
for T in [:ULogarithmic]
    @eval NiLang.AD.GVar(x::$T) = default_constructor($T, GVar(_content(x), zero(_content(x))))
    #@eval NiLang.AD.grad(x::$T{<:GVar}) = default_constructor($T, grad(_content(x)))
    @eval (_::Type{Inv{GVar}})(x::$T{<:GVar}) = default_constructor($T, (~GVar)(_content(x)))

    @eval Base.one(x::$T{GVar{T,GT}}) where {T, GT} = one($T{GVar{T,GT}})
    @eval Base.one(::Type{$T{GVar{T,GT}}}) where {T,GT} = default_constructor($T, GVar(zero(T), zero(GT)))
    @eval Base.zero(x::$T{GVar{T,GT}}) where {T,GT} =zero($T{GVar{T,GT}})
    @eval Base.zero(::Type{$T{GVar{T,T}}}) where T = default_constructor($T, GVar(zero(T), zero(T)))
end

function NiLang.loaddata(::Type{Array{<:ULogarithmic{GVar{T,T}}}}, data::Array{<:ULogarithmic{T}}) where {T}
    GVar.(data)
end

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
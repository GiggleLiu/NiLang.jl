"""
    NoGrad{T} <: IWrapper{T}
    NoGrad(x)

A `NoGrad(x)` is equivalent to `GVar^{-1}(x)`, which cancels the `GVar` wrapper.
"""
@pure_wrapper NoGrad

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
    @i function GVar(x::T) where T
        g ← zero(x)
        x ← new{T,T}(x, g)
    end

    @i function GVar(x::Integer)
    end
    @i function GVar(x::Function)
    end
    @i function GVar(x::AbstractArray)
        GVar.(x)
    end
    @i function GVar(x::Tuple)
        GVar.(x)
    end
    @i function GVar(x::NoGrad)
        (~NoGrad)(x)
    end
end
GVar(x::Complex) = Complex(GVar(x.re), GVar(x.im))
GVar(x::Complex, y::Complex) = Complex(GVar(x.re, y.re), GVar(x.im, y.im))
(_::Inv{GVar})(x::Complex) = Complex((~GVar)(x.re), (~GVar)(x.im))

Base.copy(b::GVar) = GVar(b.x, copy(b.g))
Base.zero(x::GVar) = GVar(Base.zero(x.x), Base.zero(x.g))
Base.zero(::Type{<:GVar{T}}) where T = GVar(zero(T))
Base.adjoint(b::GVar) = GVar(b.x', b.g')

# define kernel and field views
"""
    grad(var)

Get the gradient field of `var`.
"""
@fieldview grad(gv::GVar) = gv.g
@fieldview value(gv::GVar) = gv.x
# TODO: fix the problem causing this patch, the field type can not change?!
chfield(x::GVar, ::typeof(value), xval::GVar) = GVar(xval, x.g)

grad(gv::T) where T = zero(T)
grad(gv::Complex) where T = Complex(grad(gv.re), grad(gv.im))
grad(gv::AbstractArray{T}) where T = grad.(gv)
grad(gv::Function) = 0
chfield(x::T, ::typeof(grad), g::T) where T = (@invcheck iszero(g) || g≈0; x)
chfield(x::GVar, ::typeof(grad), g::GVar) where T = GVar(x.x, g)
chfield(x::Complex, ::typeof(grad), g::Complex) where T = Complex(GVar(x.re, g.re), GVar(x.im, g.im))

# NOTE: superwarning: check value only to make ancilla gradient descardable.
NiLangCore.deanc(x::GVar, val::GVar) = NiLangCore.deanc(value(x), value(val))
function NiLangCore.deanc(x::AbstractArray{<:GVar}, val::AbstractArray{<:GVar})
    for (xi, vali) in zip(x, val)
        NiLangCore.deanc(value(xi), value(vali))
    end
end

#NiLangCore.almost_same(a::GVar, b::GVar) = NiLangCore.almost_same(value(a), value(b))

# constructors and deconstructors
Base.:-(x::GVar) = GVar(-x.x, -x.g)

## variable mapping
function (_::Type{Inv{GVar}})(x::GVar{<:GVar,<:GVar})
    Partial{:x}(x)
end

Base.show(io::IO, gv::GVar) = print(io, "GVar($(gv.x), $(gv.g))")
Base.show(io::IO, ::MIME"plain/text", gv::GVar) = Base.show(io, gv)
# interfaces

"""
    Loss{T} <: IWrapper{T}
    Loss(x)

Wrapper used to mark the loss variable.
"""
@pure_wrapper Loss
grad(x::Loss) = grad(x.x)

"""
    @nograd f(args...)

Mark `f(args...)` as having no gradients.
"""
macro nograd(ex)
    @match ex begin
        :($f($(args...))) => begin
            newargs = []
            for arg in args
                push!(newargs, @match arg begin
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

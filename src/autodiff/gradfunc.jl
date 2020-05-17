export Grad, NGrad, Hessian, gradient

"""
    NGrad{N,FT} <: Function

Obtain gradients `Grad(f)(Val(i), args..., kwargs...)`, where `i` is the index of loss in `args`. `Grad` object calls forward first, and then backward.
"""
struct NGrad{N,FT} <: Function
    f::FT
end
function NGrad{N}(f::FT) where {N,FT}
    NGrad{N,FT}(f)
end

const Grad{FT} = NGrad{1,FT}
const Hessian{FT} = NGrad{2,FT}

Base.show_function(io::IO, b::NGrad{N}, compact::Bool) where {N} = print(io, "$(b.f)"*"'"^N)
Base.show_function(io::IO, ::MIME"text/plain", b::NGrad{N}, compact::Bool) where {N} = print(io, b)
Base.display(bf::NGrad) = print(bf)
(_::Type{Inv{NGrad{N}}})(f::NGrad{M}) where {M, N} = NGrad{M-N}(f.f)
(_::Type{Inv{NGrad{M}}})(f::NGrad{M}) where {M} = f.f


@i function (g::Grad)(il::Val{iloss}, args...; kwargs...) where iloss
    protectf(g).f(args...; kwargs...)
    GVar.(args)
    INC(grad(tget(args,iloss)))
    (~protectf(g).f)(args...; kwargs...)
end

@i function (g::Grad)(args...; iloss::Int, kwargs...)
    protectf(g).f(args...; kwargs...)
    GVar.(args)
    INC(grad(tget(args,iloss)))
    (~protectf(g).f)(args...; kwargs...)
end

function gradient(::Val{iloss}, f, args; kwargs...) where iloss
    args = f(args...; kwargs...)
    args = NiLangCore.wrap_tuple(GVar.(args))
    @instr INC(grad(tget(args ,iloss)))
    grad.(NiLangCore.wrap_tuple((~f)(args...; kwargs...)))
end

gradient(f, args; iloss::Int, kwargs...) = gradient(Val(iloss), f, args; kwargs...)

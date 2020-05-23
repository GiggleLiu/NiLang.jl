export jacobian, jacobian_repeat

function jacobian_repeat(f, args; kwargs...)
    jacobian_repeat(match_eltype(args), f, args; kwargs...)
end

"""
    jacobian_repeat([T], f, args...; kwargs...)

Get the Jacobian matrix for function `f(args..., kwargs...)` by computing one row (gradients) a time.
"""
function jacobian_repeat(::Type{T}, f, args; kwargs...) where T
    narg = length(args)
    res = zeros(T, narg, narg)
    for i = 1:narg
        res[i,:] .= gradient(f, args; iloss=i, kwargs...)
    end
    return res
end

function wrap_jacobian(::Type{T}, args) where T
    # get number of parameters
    N = 0
    for arg in args
        if NiLang.AD.isvar(arg)
            N += length(arg)
        end
    end

    # jacobian matrix
    jac = zeros(T, N, N)
    for i=1:N
        jac[i,i] = 1
    end

    k = 0
    res = []
    for arg in args
        if NiLang.AD.isvar(arg)
            if arg isa AbstractArray
                ri = similar(arg, GVar{T, Vector{T}})
                for l=1:length(arg)
                    k += 1
                    ri[k] = GVar(arg[l], AutoBcast(view(jac,:,k)))
                end
            else
                k += 1
                ri = GVar(arg, AutoBcast(view(jac, :, k)))
            end
            push!(res, ri)
        else
            push!(res, nothing)
        end
    end
    jac, res
end

"""
    jacobian(f, args; kwargs...)

Get the Jacobian matrix for function `f(args..., kwargs...)` by use vectorized variables in the gradient field.
"""
jacobian(f, args; kwargs...) = jacobian(Float64, f, args; kwargs...)
function jacobian(::Type{T}, f, args; kwargs...) where T
    args = f(args...; kwargs...)
    jac, args = wrap_jacobian(T, args)
    (~f)(args...; kwargs...)
    return jac
end

function match_eltype(args)
    types = Any[_eltype.(args)...]
    hasfloat = any(t->t <: AbstractFloat, types)
    hasfixed = any(t->t <: Fixed, types)
    if hasfloat && hasfixed
        error("Float point number and Fixed point numbers are not compatible!")
    end
    if hasfloat
        promote_type(filter(x->x <: AbstractFloat, types)...)
    elseif hasfixed
        promote_type(filter(x->x <: Fixed, types)...)
    else
        promote_type(types...)
    end
end

_eltype(x) = typeof(x)
_eltype(::Type{T}) where T = T
_eltype(::Type{<:AbstractArray{T}}) where T = _eltype(T)
_eltype(x::AbstractArray{T}) where T = _eltype(T)

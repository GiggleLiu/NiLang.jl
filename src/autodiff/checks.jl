export check_grad, nparams
export gradient_numeric

using FixedPointNumbers: Fixed

@nospecialize
isvar(x) = nparams(x) != 0

nparams(model) = nparams(NiLangCore.type2tuple(model))
nparams(x::AbstractArray{<:AbstractFloat}) = length(x)
nparams(x::AbstractArray{<:GVar}) = length(x)
nparams(x::AbstractArray) = sum(nparams, x)
nparams(x::Fixed) = 1
function nparams(x::Union{Tuple,NamedTuple})
    res = 0
    for xi in x
        res += nparams(xi)
    end
    res
end
nparams(x::AbstractFloat) = 1
nparams(x::GVar) = 1

function tset(vfunc::Function, tp::Tuple, iloss)
    map(i->i===iloss ? vfunc(tp[i]) : tp[i], (1:length(tp)...,))
end
function tset(value, tp::Tuple, iloss)
    map(i->i===iloss ? value : tp[i], (1:length(tp)...,))
end

function update_var(args, iarg, i::Int, val)
    args[iarg][i] += val
    args
end

function update_var(args, iarg, ::Nothing, val)
    tset(x->chfield(x, value, value(x) + val), args, iarg)
end

function ng_single(::Type{T}, f, args, kwargs, iarg, i, iloss, δ) where T
    args = update_var(args, iarg, i, T(δ/2))
    @instr f(args...; kwargs...)
    pos = value(args[iloss])
    @instr (~f)(args...; kwargs...)
    args = update_var(args, iarg, i, -T(δ))
    @instr f(args...; kwargs...)
    neg = value(args[iloss])
    @instr (~f)(args...; kwargs...)
    args = update_var(args, iarg, i, T(δ/2))
    (pos - neg)/δ
end

function ng_single(::Type{T}, f, args, kwargs, iarg, i, iloss, δ) where T<:Complex
    res = zero(T)
    for dd = [δ, im*δ]
        args = update_var(args, iarg, i, dd/2)
        @instr f(args...; kwargs...)
        pos = value(args[iloss])
        @instr (~f)(args...; kwargs...)
        args = update_var(args, iarg, i, -dd)
        @instr f(args...; kwargs...)
        neg = value(args[iloss])
        @instr (~f)(args...; kwargs...)
        args = update_var(args, iarg, i, dd/2)
        if dd == δ
            res += (pos - neg)/δ
        else
            res += im*(pos - neg)/δ
        end
    end
    res
end

function ng(f, args, iarg; iloss::Int, δ=1e-5, kwargs...)
    x = args[iarg]
    T = eltype(x)
    if x isa AbstractArray
        res = zero(x)
        for i = 1:length(x)
            res[i] = ng_single(T, f, args, kwargs, iarg, i, iloss, δ)
        end
        return res
    else
        ng_single(T, f, args, kwargs, iarg, nothing, iloss, δ)
    end
end

"""
    gradient_numeric(f, args...; iloss, kwargs...)

Numeric differentiating f(args..., kwargs...).
"""
function gradient_numeric(f, args; iloss::Int, kwargs...)
    map(1:length(args)) do iarg
        if isvar(args[iarg])
            ng(f, args, iarg; iloss=iloss, kwargs...)
        else
            0
        end
    end
end

"""
    check_grad(f, args; atol::Real=1e-8, verbose::Bool=false, iloss::Int, kwargs...)

Return true if the gradient of `f(args..., kwargs...)` is reversible.
"""
function check_grad(f, args; atol::Real=1e-4, verbose::Bool=false, iloss::Int, kwargs...)
    vars = ((iarg for iarg in 1:length(args) if isvar(args[iarg]))...,)
    initial_vars = deepcopy(vars)
    ngs = gradient_numeric(f, args; kwargs..., iloss=iloss)
    gs = gradient(f, args; kwargs..., iloss=iloss)
    verbose && @show ngs
    verbose && @show gs
    if !all(isapprox.(ngs, gs, atol=atol))
        verbose && println("gradient not match: $ngs v.s. $gs")
        return false
    end

    if !world_similar(initial_vars, vars, atol=atol, verbose=verbose)
        verbose && println("world changed during obtaining gradient.")
        return false
    end
    return true
end

@specialize

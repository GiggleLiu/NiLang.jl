export ng, check_grad
export ngradient, gradient

isvar(x) = false
isvar(x::AbstractFloat) = true
isvar(x::Loss{<:AbstractFloat}) = true
isvar(x::AbstractArray{T}) where T<:AbstractFloat = true

function tset(vfunc::Function, tp::Tuple, iloss)
    map(i->i===iloss ? vfunc(tp[i]) : tp[i], 1:length(tp))
end
function tset(value, tp::Tuple, iloss)
    map(i->i===iloss ? value : tp[i], 1:length(tp))
end

function gradient(f, args; kwargs=())
    gargs = f'(args...; kwargs...)
    return [grad(x) for x in gargs]
end

function ng(f, args, iarg, iloss; δ=1e-5, kwargs=())
    x = args[iarg]
    if x isa AbstractArray
        T = eltype(x)
        res = zero(x)
        for i = 1:length(x)
            args[iarg][i] += T(δ/2)
            @instr f(args...; kwargs...)
            pos = value(args[iloss])
            @instr (~f)(args...; kwargs...)
            args[iarg][i] -= T(δ)
            @instr f(args...; kwargs...)
            neg = value(args[iloss])
            @instr (~f)(args...; kwargs...)
            args[iarg][i] += T(δ/2)
            res[i] = (pos - neg)/δ
        end
        return res
    else
        args = tset(x->chfield(x, value, value(x) + δ/2), args, iarg)
        @instr f(args...; kwargs...)
        pos = value(args[iloss])
        @instr (~f)(args...; kwargs...)
        args = tset(x->chfield(x, value, value(x) - δ), args, iarg)
        @instr f(args...; kwargs...)
        neg = value(args[iloss])
        @instr (~f)(args...; kwargs...)
        args = tset(x->chfield(x, value, value(x) + δ/2), args, iarg)
        (pos - neg)/δ
    end
end

"""
Numeric differentiation.
"""
function ngradient(f, args; kwargs=())
    iloss = findfirst(x->x<:Loss, typeof.(args))
    @instr (~Loss)(tget(args,iloss))
    if iloss === nothing
        throw(ArgumentError("input arguments does not contain Loss! $args"))
    end
    map(1:length(args)) do iarg
        if isvar(args[iarg])
            ng(f, args, iarg, iloss; kwargs=kwargs)
        else
            0
        end
    end
end

"""
    check_grad(f, args; kwargs=(), atol::Real=1e-8, verbose::Bool=false)

Return true if the gradient of `f(args..., kwargs...)` is reversible.
"""
function check_grad(f, args; kwargs=(), atol::Real=1e-4, verbose::Bool=false)
    vars = ((iarg for iarg in 1:length(args) if isvar(args[iarg]))...,)
    initial_vars = deepcopy(vars)
    ngs = ngradient(f, args; kwargs=kwargs)
    gs = gradient(f, args; kwargs=kwargs)
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

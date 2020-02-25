export jacobian, jacobian_repeat

"""
    jacobian_repeat(f, args; kwargs=())

Get the Jacobian matrix for function `f(args..., kwargs...)` by computing one row (gradients) a time.
"""
function jacobian_repeat(f, args; kwargs=())
    narg = length(args)
    T = match_eltype(args)
    res = zeros(T, narg, narg)
    for i = 1:narg
        @instr Loss(tget(args, i))
        res[i,:] .= gradient(f, args; kwargs=kwargs)
        @instr (~Loss)(tget(args, i))
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
    jacobian(f, args; kwargs=())

Get the Jacobian matrix for function `f(args..., kwargs...)` by use vectorized variables in the gradient field.
"""
jacobian(f, args; kwargs=()) = jacobian(Float64, f, args; kwargs=kwargs)
function jacobian(::Type{T}, f, args; kwargs=()) where T
    args = f(args...; kwargs...)
    jac, args = wrap_jacobian(T, args)
    (~f)(args...; kwargs...)
    return jac
end

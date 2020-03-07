export hessian_backback

@i function backback(f, args...; index::Int, iloss::Int, kwargs...)
    # forward
    Grad(f)(args...; kwargs..., iloss=iloss)

    for i = 1:length(args)
        GVar(grad(tget(args,i)))
        GVar(value(tget(args,i)))
    end
    grad(grad(tget(args,index))) ⊕ 1
    # backward#2
    (~Grad(f))(args...; kwargs..., iloss=iloss)
end

"""
    hessian_backback(f, args; iloss::Int, kwargs...)

Obtain the Hessian matrix of `f(args..., kwargs...)` by back propagating adjoint program.
"""
function hessian_backback(f, args; iloss::Int, kwargs...)
    N = length(args)
    hmat = zeros(N, N)
    for i=1:N
        if !(args[i] isa Integer || args[i] isa AbstractVector)
            res = backback(f, args...; kwargs..., index=i, iloss=iloss)
            hmat[:,i] .= map(x->grad(value(x)), res[2:end])
        end
    end
    hmat
end

function hessian_numeric(f, args; iloss::Int, η=1e-5, kwargs...)
    narg = length(args)
    res = zeros(narg, narg)
    largs = [args...]
    for i = 1:narg
        if nparams(args[i]) == 1
            @instr value(largs[i]) ⊕ η/2
            gpos = gradient(f, largs; iloss=iloss, kwargs...)
            @instr value(largs[i]) ⊖ η
            gneg = gradient(f, largs; iloss=iloss, kwargs...)
            @instr value(largs[i]) ⊕ η/2
            res[:,i] .= (gpos .- gneg)./η
        end
    end
    return res
end

function local_hessian_numeric(f, args; kwargs...)
    nargs = length(args)
    hes = zeros(nargs,nargs,nargs)
    for j=1:nargs
        if nparams(args[j]) == 1
            hes[:,:,j] .= hessian_numeric(f, args; kwargs..., iloss=j)
        end
    end
    mask = BitArray(nparams.(args) .== 1)
    hes[mask, mask, mask]
end

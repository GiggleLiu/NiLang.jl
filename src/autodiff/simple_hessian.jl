export simple_hessian, nhessian, local_nhessian

# TODEP
#@i function ⊕(*)(out!::Partial, x::Partial, y::Partial)
#    ⊕(*)(value(out!), value(x), value(y))
#end

#@i function ⊕(identity)(out!::Partial, x::Partial)
#    ⊕(identity)(value(out!), value(x))
#end

@i function hessian1(f, args; kwargs, index::Int)
    @safe @assert count(x -> x isa Loss, args) == 1
    iloss ← 0
    @routine begin
        for i=1:length(args)
            if (tget(args,i) isa Loss, iloss==i)
                iloss += identity(i)
            end
        end
    end

    # forward
    Grad(f)(args...; kwargs...)

    (~Loss)(tget(args,iloss))
    for i = 1:length(args)
        GVar(grad(tget(args,i)))
        GVar(value(tget(args,i)))
    end
    grad(grad(tget(args,index))) ⊕ 1
    # backward#2
    (Loss)(tget(args,iloss))
    (~f')(args...; kwargs...)

    ~@routine
end

"""
    simple_hessian(f, args::Tuple; kwargs=())

Obtain the Hessian matrix of `f(args..., kwargs...)` by differentiating the first order gradients.
"""
function simple_hessian(f, args::Tuple; kwargs=())
    N = length(args)
    hmat = zeros(N, N)
    for i=1:N
        if !(args[i] isa Integer || args[i] isa AbstractVector)
            res = hessian1(f, args; kwargs=kwargs, index=i)
            hmat[:,i] .= map(x->x isa Loss ? grad(value(value(x))) :  grad(value(x)), res[2])
        end
    end
    hmat
end

function nhessian(f, args; kwargs=(), η=1e-5)
    largs = Any[args...]
    narg = length(largs)
    res = zeros(narg, narg)
    for i = 1:narg
        if nparams(args[i]) == 1
            @instr value(largs[i]) ⊕ η/2
            gpos = gradient(f, (largs...,); kwargs=kwargs)
            @instr value(largs[i]) ⊖ η
            gneg = gradient(f, (largs...,); kwargs=kwargs)
            @instr value(largs[i]) ⊕ η/2
            res[:,i] .= (gpos .- gneg)./η
        end
    end
    return res
end

function local_nhessian(f, args; kwargs=())
    nargs = length(args)
    hes = zeros(nargs,nargs,nargs)
    for j=1:nargs
        if nparams(args[j]) == 1
            @instr Loss(tget(args, j))
            hes[:,:,j] .= nhessian(f, args; kwargs=kwargs)
            @instr (~Loss)(tget(args, j))
        end
    end
    mask = BitArray(nparams.(args) .== 1)
    hes[mask, mask, mask]
end

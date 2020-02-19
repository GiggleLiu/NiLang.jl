export simple_hessian, nhessian, simple_jacobian, jacobian, local_nhessian

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
        if !(args[i] isa Integer || args[i] isa AbstractVector)
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
        @instr Loss(tget(args, j))
        hes[:,:,j] .= nhessian(f, args; kwargs=kwargs)
        @instr (~Loss)(tget(args, j))
    end
    hes
end

"""
    simple_jacobian(f, args; kwargs=())

Get the Jacobian matrix for function `f(args..., kwargs...)`.
"""
function simple_jacobian(f, args; kwargs=())
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
        if isvar(arg)
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
        if isvar(arg)
            if arg isa AbstractArray
                ri = similar(arg, GVar{T, Vector{T}})
                for l=1:length(arg)
                    k += 1
                    ri[k] = GVar(arg[l], view(jac,:,k))
                end
            else
                k += 1
                ri = GVar(arg, view(jac, :, k))
            end
            push!(res, ri)
        end
    end
    jac, res
end

function jacobian(f, args; kwargs=())
    args = f(args...; kwargs...)
    jac, args = wrap_jacobian(Float64, args)
    (~f)(args...; kwargs...)
    return jac
end

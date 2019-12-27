export simple_hessian, nhessian

function (_::Type{Inv{GVar}})(x::GVar{<:GVar,<:GVar})
    Partial{:x}(x)
end

@i function ⊕(*)(out!::Partial, x::Partial, y::Partial)
    ⊕(*)(value(out!), value(x), value(y))
end

@i function ⊕(identity)(out!::Partial, x::Partial)
    ⊕(identity)(value(out!), value(x))
end

@i function hessian1(f, args; kwargs, index::Int)
    @safe @assert count(x -> x isa Loss, args) == 1
    @anc iloss = 0
    @routine getiloss begin
        for i=1:length(args)
            if (args[i] isa Loss, iloss==i)
                iloss += identity(i)
            end
        end
    end

    # forward
    f'(args...; kwargs...)

    (~Loss)(args[iloss])
    for i = 1:length(args)
        GVar(grad(args[i]))
        GVar(value(args[i]))
    end
    grad(grad(args[index])) ⊕ 1.0
    # backward#2
    (Loss)(args[iloss])
    (~f')(args...; kwargs...)

    ~@routine getiloss
end

function simple_hessian(f, args::Tuple; kwargs=())
    N = length(args)
    hmat = zeros(N, N)
    for i=1:N
        res = hessian1(f, args; kwargs=kwargs, index=i)
        hmat[:,i] .= map(x->x isa Loss ? grad(value(value(x))) :  grad(value(x)), res[2])
    end
    hmat
end

function nhessian(f, args; kwargs=(), η=1e-5)
    largs = Any[args...]
    narg = length(largs)
    res = zeros(narg, narg)
    for i = 1:narg
        @instr value(largs[i]) ⊕ η/2
        gpos = gradient(f, (largs...,); kwargs=kwargs)
        @instr value(largs[i]) ⊖ η
        gneg = gradient(f, (largs...,); kwargs=kwargs)
        @instr value(largs[i]) ⊕ η/2
        res[:,i] .= (gpos .- gneg)./η
    end
    return res
end

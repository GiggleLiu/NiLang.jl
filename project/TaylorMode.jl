using TaylorSeries
using NiLang

x, y = set_variables("x y", order=3)
z = x + y

function nhessian(f, args; kwargs=(), η=1e-5)
    largs = Any[args...]
    narg = length(largs)
    res = zeros(narg, narg)
    for i = 1:narg
        @instr val(largs[i]) += η/2
        gpos = gradient(f, (largs...,); kwargs=kwargs)
        @instr val(largs[i]) -= η
        gneg = gradient(f, (largs...,); kwargs=kwargs)
        @instr val(largs[i]) += η/2
        res[:,i] .= (gpos .- gneg)./η
    end
    return res
end

function hessian(f, args; kwargs=(), η=1e-5)
    largs = Any[args...]
    narg = length(largs)
    res = zeros(narg, narg)
    for i = 1:narg
        @instr val(largs[i]) += η/2
        gpos = gradient(f, (largs...,); kwargs=kwargs)
        @instr val(largs[i]) -= η
        gneg = gradient(f, (largs...,); kwargs=kwargs)
        @instr val(largs[i]) += η/2
        res[:,i] .= (gpos .- gneg)./η
    end
    return res
end

g = HeissianGrid(...)

@i function ⊕(*)(out!::TaylorVar{N}, a::TaylorVar{N}, b::TaylorVar{N}) where N
    ⊕(*)(out!.x, a.x, b.x)
    global_tape[a.index] .*= b.x
    global_tape[a.index] .*= b.x
    global_tape[b.index] .*= a.x
end


nhessian(⊕(*), (Loss(0.0), 1.0, 2.0))

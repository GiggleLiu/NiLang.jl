export bennett, bennett!

function direct_emulate(step, x0::T, args...; N::Int, kwargs...) where T
    xpre = copy(x0)
    local x
    for i=1:N
        x = zero(xpre)
        res = step(x, xpre, args...; kwargs...)
        xpre = res[1]
        args = res[3:end]
    end
    return xpre
end

struct BennettLog
    fcalls::Vector{NTuple{3,Any}}  # depth, function index f_i := s_{i-1} -> s_{i}, length should be `(2k-1)^n` and function
    peak_mem::Base.RefValue{Int}  # should be `n*(k-1)+2`
    depth::Base.RefValue{Int}
end
BennettLog() = BennettLog(NTuple{3,Any}[], Ref(0), Ref(0))

# hacking the reversible program
function logfcall(l::BennettLog, i, f)
    push!(l.fcalls, (l.depth[], i, f))
    l, i, f
end
function ilogfcall(l::BennettLog, i, f)
    push!(l.fcalls, (l.depth[], i, ~f))
    l, i, f
end

@dual logfcall ilogfcall

Base.show(io::IO, ::MIME"text/plain", logger::BennettLog) = Base.show(io, logger)
function Base.show(io::IO, logger::BennettLog)
    nreverse = count(x->x[3] isa Inv, logger.fcalls)
    print(io, """Bennett log
| peak memory usage = $(logger.peak_mem[])
| number of function forward/backward calls = $(length(logger.fcalls)-nreverse)/$nreverse""")
end

"""
    bennett(step, y, x, args...; k, N, logger=BennettLog(), kwargs...)

* `step` is a reversible step function,
* `y` is the output state,
* `x` is the input state,
* `k` is the number of steps in each Bennett's recursion,
* `N` is the total number of steps,
* `logger=BennettLog()` is the logging of Bennett's algorithm,
* `args...` and `kwargs...` are additional arguments for steps.
"""
@i function bennett(step, y::T, x::T, args...; k::Int, N::Int, logger=BennettLog(), kwargs...) where T
    state ← Dict{Int, T}()
    state[1] ← zero(x)
    state[1] +=  x
    bennett!((@skip! step), state, k, 1, N, args...; do_uncomputing=true, logger=logger, kwargs...)
    SWAP(y, state[N+1])
    state[1] -= x
    state[1] → zero(x)
    state[N+1] → zero(x)
    state → Dict{Int, T}()
end

"""
    bennett!(step, state::Dict, args...; k, N, logger=BennettLog(), do_uncomputing=false, kwargs...)

* `step` is a reversible step function,
* `state` is the dictionary state, with `state[1]` the input state, the return value is stored in `state[N+1]`,
* `k` is the number of steps in each Bennett's recursion,
* `N` is the total number of steps,
* `logger=BennettLog()` is the logging of Bennett's algorithm,
* `args...` and `kwargs...` are additional arguments for steps.
"""
@i function bennett!(step, state::Dict{Int,T}, args...; k::Int, N::Int, logger=BennettLog(), do_uncomputing=false, kwargs...) where T
    bennett!(step, state, k, 1, N, args...; logger=logger, do_uncomputing=do_uncomputing, kwargs...)
end

@i function bennett!(step, state::Dict{Int,T}, k::Int, base, len, args...; logger, do_uncomputing, kwargs...) where T
    @safe logger.depth[] += 1
    @invcheckoff if len == 1
        state[base+1] ← zero(state[base])
        @safe logger.peak_mem[] = max(logger.peak_mem[], length(state))
        step(state[base+1], state[base], args...; kwargs...)
        logfcall(logger, (@const base+1), step)
    else
        @routine begin
            @zeros Int nstep n
            n += ceil((@skip! Int), (@const len / k))
            nstep += ceil((@skip! Int), (@const len / n))
        end
        for j=1:nstep
            bennett!(step, state, k, (@const base+n*(j-1)), (@const min(n,len-n*(j-1))), args...; logger=logger, do_uncomputing=true, kwargs...)
        end
        if do_uncomputing
            for j=nstep-1:-1:1
                ~bennett!(step, state, k, (@const base+n*(j-1)), n, args...; logger=logger, do_uncomputing=true, kwargs...)
            end
        end
        ~@routine
    end
end

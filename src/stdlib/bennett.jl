export bennett

@i function bennett(step, state::Dict{Int,T}, k::Int, base, len, args...; kwargs...) where T
    @invcheckoff if len == 1
        state[base+1] ← zero(state[base])
        step(state[base+1], state[base], args...; kwargs...)
    else
        @routine begin
            @zeros Int nstep n
            n += ceil((@skip! Int), (@const len / k))
            nstep += ceil((@skip! Int), (@const len / n))
        end
        for j=1:nstep
            bennett(step, state, k, (@const base+n*(j-1)), (@const min(n,len-n*(j-1))), args...; kwargs...)
        end
        for j=nstep-1:-1:1
            ~bennett(step, state, k, (@const base+n*(j-1)), n, args...; kwargs...)
        end
        ~@routine
    end
end

@i function i_addto!(target::AbstractArray, source::AbstractArray)
    @safe @assert length(target) == length(source)
    @inbounds for i=1:length(target)
        target[i] += source[i]
    end
end

@i @inline function i_addto!(target, source)
    target += source
end

@i function bennett(step, y::T, x::T, args...; k::Int, nsteps::Int, kwargs...) where T
    state ← Dict{Int, T}()
    state[1] ← zero(x)
    i_addto!(state[1], x)
    bennett((@skip! step), state, k, 1, nsteps, args...; kwargs...)
    SWAP(y, state[nsteps+1])
    ~i_addto!(state[1], x)
    state[1] → zero(x)
    state[nsteps+1] → zero(x)
    state → Dict{Int, T}()
end

function direct_emulate(step, x0::T, args...; nsteps::Int, kwargs...) where T
    xpre = copy(x0)
    local x
    for i=1:nsteps
        x = zero(xpre)
        res = step(x, xpre, args...; kwargs...)
        xpre = res[1]
        args = res[3:end]
    end
    return xpre
end
"""
solve the 1D linear advection equation
```math
∂q/∂t=−u∂q/∂x
```
in a periodic domain, where ``q`` is the quantity being advected,
``t`` is time, ``x`` is the spatial coordinate and ``u`` is the velocity,
which is constant with ``x``. 
"""
function lax_wendroff!(nt::Int, c, q_init::AbstractVector{T}, q::AbstractVector{T}) where T
    nx = length(q)
    flux = zeros(T, nx-1)   # Fluxes between boxes
    @inbounds for i=1:nx
        q[i] = q_init[i] # Initialize q
    end
    @inbounds for j=1:nt  # Main loop in time
        for i=1:nx-1
            flux[i] = 0.5*c*(q[i]+q[i+1]+c*(q[i]-q[i+1]))
        end
        for i=2:nx-1
            q[i] += flux[i-1]-flux[i]
        end
        q[1] = q[nx-1]; q[nx] = q[2] # Treat boundary conditions
    end
    return q
end

using Random
Random.seed!(2)
q_init = randn(100)
q = zeros(100)
@show lax_wendroff!(2000, 1.0, q_init, zero(q_init))
using BenchmarkTools
@benchmark lax_wendroff!(2000, 1.0, $q_init, x) setup=(x=zero(q_init))
@time lax_wendroff!(2000, 1.0, q_init, q)

using NiLang
@i function i_lax_wendroff!(nt::Int, c, q_init::AbstractVector{T}, q::AbstractVector{T},
        cache::AbstractMatrix{T}) where T
    nx ← length(q)
    @inbounds for i=1:nx
        q[i] += q_init[i] # Initialize q
    end
    @inbounds for j=1:nt  # Main loop in time
        for i=1:nx-1
            @routine begin
                @zeros T anc1 anc2 anc3
                anc1 += 0.5 * c
                anc2 += q[i] - q[i+1]
                anc3 += q[i] + q[i+1]
                anc3 += c * anc2
            end
            cache[i,j] += anc1 * anc3
            ~@routine
        end
        for i=2:nx-1
            q[i] += cache[i-1,j]-cache[i,j]
        end
        # Treat boundary conditions
        cache[nx,j] += q[nx-1]
        SWAP(q[1], cache[nx,j])
        cache[nx+1,j] += q[2]
        SWAP(q[nx], cache[nx+1,j])
    end
end
nt = 2000
i_lax_wendroff!(nt, 1.0, q_init, zero(q_init), zeros(length(q_init)+1,nt))

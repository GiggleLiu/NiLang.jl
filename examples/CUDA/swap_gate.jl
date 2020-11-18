using CuArrays, CUDAnative, GPUArrays
using NiLang, NiLang.AD

"""
A reversible swap kernel for GPU for SWAP gate in quantum computing.
See the irreversible version for comparison

http://tutorials.yaoquantum.org/dev/generated/developer-guide/2.cuda-acceleration/
"""
@i @inline function swap_kernel(state::AbstractVector{T}, mask1, mask2) where T
    @invcheckoff b ← (blockIdx().x-1) * blockDim().x + threadIdx().x
    @invcheckoff if (b < length(state), ~)
        if (b&mask1==0 && b&mask2==mask2, ~)
            NiLang.SWAP(state[b+1], state[b ⊻ (mask1|mask2) + 1])
        end
    end
end

# TODO: support ::Type like argument.
"""
SWAP gate in quantum computing.
"""
@i function instruct!(state::CuVector, gate::Val{:SWAP}, locs::Tuple{Int,Int})
    mask1 ← 1 << (locs[1]-1)
    mask2 ← 1 << (locs[2]-1)
    XY ← GPUArrays.thread_blocks_heuristic(length(state))
    @cuda threads=XY[1] blocks=XY[2] swap_kernel(state, mask1, mask2)
end

using Test
@testset "swap gate" begin
    v = cu(randn(128))
    v1 = instruct!(copy(v), Val(:SWAP), (3,4))[1]
    v2 = instruct!(copy(v1), Val(:SWAP), (3,4))[1]
    v3 = (~instruct!)(copy(v1), Val(:SWAP), (3,4))[1]
    @test !(v ≈ v1)
    @test v ≈ v2
    @test v ≈ v3
end

@i function loss(out!, state::CuVector)
    instruct!(state, Val(:SWAP), (3,4))
    out! += state[4]
end

loss(0.0, CuArray(randn(128)))
Grad(loss)(Loss(0.0), CuArray(randn(128)))

####################### A different loss ###############
@i function loss(out!, state::CuVector, target::CuVector)
    instruct!(state, Val(:SWAP), (3,4))
    out! += state' * target
end

# requires defining a new primitive, we don't how to parallelize a CUDA program automatically yet.
using LinearAlgebra: Adjoint
function (_::MinusEq{typeof(*)})(out!::GVar, x::Adjoint{<:Any, <:CuVector{<:GVar}}, y::CuVector{<:GVar})
    chfield(out!, value, value(out!)-(value.(x) * value.(y))[]),
    chfield.(parent(x), grad, grad.(parent(x)) .+ grad(out!)' .* conj.(value.(y)))',
    chfield.(y, grad, grad.(y) .+ grad(out!) .* conj.(value.(x')))
end

function (_::PlusEq{typeof(*)})(out!::GVar, x::Adjoint{<:Any, <:CuVector{<:GVar}}, y::CuVector{<:GVar})
    chfield(out!, value, value(out!)+(value.(x) * value.(y))[]),
    chfield.(parent(x), grad, grad.(parent(x)) .- grad(out!)' .* conj.(value.(y)))',
    chfield.(y, grad, grad.(y) .- grad(out!) .* conj.(value.(x')))
end

function (_::PlusEq{typeof(*)})(out!, x, y)
    out! += x * y
    out!, x, y
end

function (_::MinusEq{typeof(*)})(out!, x, y)
    out! -= x * y
    out!, x, y
end

loss(0.0, CuArray(randn(128)), CuArray(randn(128)))
loss'(Loss(0.0), CuArray(randn(128)), CuArray(randn(128)))

\begin{minipage}{.88\columnwidth}
\begin{lstlisting}
using CuArrays, CUDAnative, GPUArrays
using NiLang, NiLang.AD

@i @inline function swap_kernel(state, mask1, mask2)
    @invcheckoff begin
        b ← (blockIdx().x-1) * 
            blockDim().x + threadIdx().x - 1
        if (b < length(state), ~)
            if (b&mask1==0 && b&mask2==mask2, ~)
                SWAP(state[b+1], state[b ⊻ 
                    (mask1|mask2) + 1])
            end
        end
        b → (blockIdx().x-1) * 
            blockDim().x + threadIdx().x - 1
    end
end
\end{lstlisting}
\end{minipage}

This kernel function simulates the SWAP gate in quantum computing.
Here, one must use the macro \texttt{@invcheckoff} to turn off the reversibility checks. It is necessary because the possible error thrown in a kernel function can not be handled on a CUDA kernel.
One can launch this kernel function to GPUs with a single macro \texttt{@cuda}, as shown in the following using case.

\begin{minipage}{\columnwidth}
\begin{lstlisting}[multicols=2]
julia> @i function instruct!(state::CuVector,
            gate::Val{:SWAP}, locs::Tuple{Int,Int})
           mask1 ← 1 << (tget(locs, 1)-1)
           mask2 ← 1 << (tget(locs, 2)-1)
           XY ← GPUArrays.thread_blocks_heuristic(
                length(state))
           @cuda threads=tget(XY,1) blocks=tget(XY,
                2) swap_kernel(state, mask1, mask2)
       end

julia> instruct!(CuArray(randn(8)),
            Val(:SWAP), (1,3))[1]
8-element CuArray{Float64,1,Nothing}:
 -0.06956048379200473
 -0.6464176838567472
 -0.06523362834285944
 -0.7314356941903547
  1.512329204247244
  0.9773772766637732
  1.6473223915215722
 -1.0631789613639087
\end{lstlisting}
\end{minipage}

One can also write kernels with KernelAbstaction. It solves many compatibility issues related to different function calls on GPU and CPU.


\begin{minipage}{.88\columnwidth}
\begin{lstlisting}
@i @kernel function swap_kernel2(state, mask1, mask2)
    @invcheckoff begin
        b ← @index(Global)
        if (b < length(state), ~)
            if (b&mask1==0 && b&mask2==mask2, ~)
                SWAP(state[b+1], state[b ⊻ 
                    (mask1|mask2) + 1])
            end
        end
        b → @index(Global)
    end
end
\end{lstlisting}
\end{minipage}

We can use the macro \texttt{@launchkernel} to launch a kernel.
The first parameter is a device.
The second parameter is the block size.
The third parameter is the number of threads.
The last parameter is a kernel function call to be launched.

\begin{minipage}{\columnwidth}
\begin{lstlisting}[multicols=2]
julia> @i function instruct!(state::CuVector,
            gate::Val{:SWAP}, locs::Tuple{Int,Int})
           mask1 ← 1 << (tget(locs, 1)-1)
           mask2 ← 1 << (tget(locs, 2)-1)
           XY ← GPUArrays.thread_blocks_heuristic(
                length(state))
           @launchkernel CUDA() 256 length(out!
                ) swap_kernel2(state, mask1, mask2)
       end

julia> instruct!(CuArray(randn(8)),
            Val(:SWAP), (1,3))[1]
8-element CuArray{Float64,1,Nothing}:
  2.1492759883720525 
  2.326837084303501  
  1.4587667131427016 
 -1.3273806428138293 
 -0.03975355575683114
 -0.10763082744447787
 -1.7111718557581195 
 -0.47922613687722704
\end{lstlisting}
\end{minipage}




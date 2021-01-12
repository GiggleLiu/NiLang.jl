using CUDA, GPUArrays
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
    @cuda threads=256 blocks=ceil(Int,length(state)/256) swap_kernel(state, mask1, mask2)
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
Grad(loss)(Val(1), 0.0, CuArray(randn(128)))

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
Grad(loss)(Val(1), 0.0, CuArray(randn(128)), CuArray(randn(128)))
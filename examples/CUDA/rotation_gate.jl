using CUDA, GPUArrays
using NiLang, NiLang.AD

const RotGates = Union{Val{:Rz}, Val{:Rx}, Val{:Ry}}

@i @inline function instruct!(state::CuVector, gate::RotGates, loc::Int, theta::Real)
    mask ← 1<<(loc-1)
    @cuda threads=256 blocks=ceil(Int, length(state)/256) rot_kernel(gate, state, mask, theta)
end
#     @launchkernel CUDADevice() 256 length(out!) bessel_kernel(out!, v, z)

@i @inline function rot_kernel(gate::Val{:Rz}, state, mask, θ)
    @invcheckoff b ← (blockIdx().x-1) * blockDim().x + threadIdx().x
    @invcheckoff if (b < length(state) && b & mask == 0, ~)
        ROT_INSTRUCT(gate, state[b+1], state[b⊻mask+1], θ)
    end
end

@i @inline function ROT_INSTRUCT(gate::Val{:Rz}, a::T, b, θ) where T
    # make sure `invcheck` is turned off!
    @routine @invcheckoff begin
        @zeros T anc1 anc2 anc3 anc4
        anc1 += θ*(0.5im)
        anc2 += CUDA.exp(anc1)
    end
    anc3 += a * anc2'
    anc4 += b * anc2
    NiLang.SWAP(a, anc3)
    NiLang.SWAP(b, anc4)
    anc3 -= a / anc2'
    anc4 -= b / anc2
    ~@routine
end

v = randn(ComplexF64, 128) |> CuArray
v1 = instruct!(copy(v), Val(:Rz), 3, 0.5)[1]
# we can not obtain the gradient for the race condition.


# TODO: Rx and Ry gates, not finished!
@i @inline function ROT_INSTRUCT(gate::Val{:Rx}, a, b, θ)
    ROT_INSTRUCT(Val(:Rz), a, b, π/2)
    ROT_INSTRUCT(Val(:Ry), a, b, θ)
    ROT_INSTRUCT(Val(:Rz), a, b, -π/2)
end

@i @inline function ROT_INSTRUCT(gate::Val{:Ry}, a, b, θ)
    divint(θ, 2)
    ROT(a, b, θ)
    mulint(θ, 2)
end

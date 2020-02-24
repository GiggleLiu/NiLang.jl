@i @inline function instruct!(state::CuVector, gate::Val{:Rx}, loc::Int, theta::Real)
    XY ← GPUArrays.thread_blocks_heuristic(length(state))
    @routine begin
        @invcheckoff mask ← 1<<(loc-1)
    end
    @cuda threads=tget(XY,1) blocks=tget(XY,2) rx_kernel(state, mask, theta)
    ~@routine
end

@i @inline function rx_kernel(state, mask, θ)
    @routine begin
        @invcheckoff b ← (blockIdx().x-1) * blockDim().x + threadIdx().x
    end
    @invcheckoff if (b < length(state) && b & mask == 0, ~)
        RX_INSTRUCT(state[b+1], state[b⊻mask+1], θ)
    end
end

@i @inline function rz_kernel(state, mask, θ)
    @routine begin
        @invcheckoff b ← (blockIdx().x-1) * blockDim().x + threadIdx().x
    end
    @invcheckoff if (b < length(state) && b & mask == 0, ~)
        RZ_INSTRUCT(state[b+1], state[b⊻mask+1], θ)
    end
end

@i @inline function RX_INSTRUCT(a, b, θ)
    RZ_INSTRUCT(a, b, π/2)
    RY_INSTRUCT(a, b, θ)
    RZ_INSTRUCT(a, b, -π/2)
end

@i @inline function RY_INSTRUCT(a, b, θ)
    DIVINT(θ, 2)
    ROT(a, b, θ)
    MULINT(θ, 2)
end

@i @inline function RZ_INSTRUCT(a, b, θ)
    @routine @invcheckoff begin
        anc1 ← zero(a)
        anc2 ← zero(a)
        anc3 ← zero(a)
        anc4 ← zero(a)
        anc1 += θ*(0.5im)
        anc2 += CUDAnative.exp(anc1)
    end
    anc3 += a * anc2'
    anc4 += b * anc2
    NiLang.SWAP(a, anc3)
    NiLang.SWAP(b, anc4)
    anc3 -= a / anc2'
    anc4 -= b / anc2
    ~@routine
end

#all(RZ_INSTRUCT(1.0, 3.0, π) .≈ (-1.0im, 3.0im, π))
#all(RX_INSTRUCT(1.0, 3.0, π) .≈ (-3.0im, -1.0im, π))

v = randn(ComplexF64, 128) |> CuArray
v1 = instruct!(copy(v), Val(:Rx), 3, 0.5)[1]

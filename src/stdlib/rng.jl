using Random

export LcgRNG

struct LcgRNG{T<:Integer} <: AbstractRNG
    a::T
    c::T
    m::T
    x::T
end
LcgRNG(; seed=42) = LcgRNG(
    UInt(6364136223846793005),
    UInt(1442695040888963407),
    UInt(1)<<UInt(63),
    UInt(seed),
)

# https://stackoverflow.com/questions/2911432/reversible-pseudo-random-sequence-generator
# https://github.com/bobbaluba/rlcg/blob/master/include/rlcg.hpp
@i function Base.rand(rng::LcgRNG{T}) where T
    xnew ← zero(T)
    @routine begin
        anc ← zero(T)
        anc += rng.a * rng.x
        anc += rng.c
    end
    xnew += mod(anc, rng.m)
    ~@routine

    # uncompute x
    @routine begin
        @zeros T x_c anc1 ainv
        ainv += gcdx(rng.a, rng.m) |> tget(2)
        x_c += xnew - rng.c
        anc1 += ainv * x_c
    end
    rng.x -= mod(anc1, rng.m)
    ~@routine
    SWAP(rng.x, xnew)
    xnew → zero(T)
end
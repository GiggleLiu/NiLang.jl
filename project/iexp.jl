using NiLang, NiLang.AD

@i function iexp(out!, x::T; atol::Float64=1e-14) where T
    @anc anc1 = zero(x)
    @anc anc2 = zero(x)
    @anc anc3 = zero(x)
    @anc iplus = 0

    out! ⊕ 1.0
    anc1 ⊕ 1.0
    while (value(anc1) > atol, iplus != 0)
        iplus ⊕ 1
        anc2 += anc1 * x
        anc3 += anc2 / iplus
        out! ⊕ anc3
        # speudo inverse
        anc1 -= anc2 / x
        anc2 -= anc3 * iplus
        SWAP(anc1, anc3)
    end

    ~(while (value(anc1) > atol, iplus != 0)
        iplus ⊕ 1
        anc2 += anc1 * x
        anc3 += anc2 / iplus
        # speudo inverse
        anc1 -= anc2 / x
        anc2 -= anc3 * iplus
        SWAP(anc1, anc3)
    end)
    anc1 ⊖ 1.0
end

using Test
# NOTE: to allow high performance use of f += T(x).
# Now, this kind of use is a performance killer.
@testset "iexp" begin
    out = 0.0
    x = 1.3
    @instr iexp(out, x)
    res = exp(x)
    @test check_inv(iexp,(out, x))
    @test out ≈ res

    out = 0.0
    x = 1e-9
    @instr iexp(out, x)
    res = exp(x)
    @test check_inv(iexp,(out, x))
    @test out ≈ res

    out = 0.0
    x = 1.0

    @instr iexp(out, x)
    res = exp(x)
    @test check_inv(iexp,(out, x))
    @test out ≈ res
end

@testset "iexp grad" begin
    NiLangCore.GLOBAL_ATOL[] = 1e-2
    out = 0.0
    x = 1.6
    gres = exp(x)
    @test check_inv(iexp, (out, x); verbose=true)
    @test check_grad(iexp, (Loss(out), x); verbose=true)

    out = 0.0
    x = 1.6
    @instr iexp'(Loss(out), x)
    @test grad(x) ≈ gres

    #h1 = taylor_hessian(iexp, (Loss(0.0), 1.6))
    h2 = simple_hessian(iexp, (Loss(0.0), 1.0))
    nh = nhessian(iexp, (Loss(0.0), 1.0))
    #@test h1 ≈ nh
    @test isapprox(h2, nh, atol=1e-3)
end

function Base.zero(x::BeijingRing{T}) where T
    zero(BeijingRing{T})
end

function Base.zero(x::Type{BeijingRing{T}}) where T
    beijingring(zero(T))
end

function Base.zero(x::T) where T<:Partial
    zero(T)
end

function Base.zero(x::Type{<:Partial{FIELD,T}}) where {FIELD, T}
    Partial{FIELD}(Base.zero(T))
end

function Base.one(x::T) where T<:Partial
    one(T)
end

function Base.one(x::Type{<:Partial{FIELD,T}}) where {FIELD, T}
    Partial{FIELD}(Base.one(T))
end

function Base.one(x::T) where T<:GVar
    one(T)
end

function Base.one(x::Type{<:GVar{T}}) where {T}
    GVar(Base.one(T))
end

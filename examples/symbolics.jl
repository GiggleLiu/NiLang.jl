using NiLang, NiLang.AD
using SymbolicUtils
using SymbolicUtils: Term, Sym
using LinearAlgebra

const SymReal = Sym{Real}
const TermReal = Term{Real}
const SReals = Union{Term{Real}, Sym{Real}}

import NiLang: NEG, INC, DEC, ROT, IROT, FLIP
@inline function NEG(a!::SReals)
    -a!
end
@inline FLIP(b::Sym{Bool}) = !b

@inline function INC(a!::SReals)
    a! + one(a!)
end

@inline function DEC(a!::SReals)
    a! - one(a!)
end

@inline function ROT(i::SReals, j::SReals, θ::SReals)
    a, b = rot(i, j, θ)
    a, b, θ
end

@inline function IROT(i::SReals, j::SReals, θ::SReals)
    i, j, _ = ROT(i, j, -θ)
    i, j, θ
end

NiLang.AD.GVar(x::SReals) = NiLang.AD.GVar(x, zero(x))
Base.convert(::Type{SymReal}, x::Integer) = SymReal(Symbol(x))
Base.convert(::Type{Term{Real}}, x::Integer) = TermReal(Symbol(x))

Base.zero(x::Sym{T}) where T = zero(Sym{T})
Base.one(x::Sym{T}) where T = one(Sym{T})
Base.zero(::Type{<:Sym{T}}) where T = Sym{T}(Symbol(0))
Base.zero(::Type{<:Term{T}}) where T = Term{T}(Symbol(0))
Base.one(::Type{<:Sym{T}}) where T = Sym{T}(Symbol(1))
Base.one(::Type{<:Term{T}}) where T = Term{T}(Symbol(1))
Base.iszero(x::Sym{T}) where T = x === zero(x)
Base.adjoint(x::SReals) = x
SymbolicUtils.Term{T}(x::Sym{T}) where T = Term{T}(x.name)

LinearAlgebra.dot(a::T, b::T) where T<:SReals = a * b

include("sparse.jl")

using BenchmarkTools, Random
syms = @syms a::Real b::Real c::Real d::Real e::Real f::Real g::Real
Base.rand(r::Random.AbstractRNG, ::Type{SymReal}, i::Integer) = rand(r, syms, i)
Base.rand(r::Random.AbstractRNG, ::Type{TermReal}, i::Integer) = rand(r, TermReal.(syms), i)
a = sprand(TermReal, 100, 100, 0.05);
b = sprand(TermReal, 100, 100, 0.05);
@benchmark SparseArrays.dot($a, $b)
@benchmark idot(TermReal(Symbol(0)), $a, $b)
@benchmark Grad(idot)(Val(1), TermReal(Symbol(0)), $a, $b)
GVar(a)

include("Symbolics/symlib.jl")
syms = @vars a b c d e f g
Base.rand(r::Random.AbstractRNG, ::Type{<:Basic}, i::Integer) = rand(r, syms, i)
a = sprand(Basic, 100, 100, 0.05);
b = sprand(Basic, 100, 100, 0.05);
@benchmark SparseArrays.dot($a, $b)
@benchmark idot(Basic(0), $a, $b)
@benchmark Grad(idot)(Val(1), Basic(0), $a, $b)

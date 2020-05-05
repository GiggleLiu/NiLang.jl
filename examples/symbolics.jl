using NiLang, NiLang.AD
using SymbolicUtils

NiLang.AD.GVar(x::SymbolicUtils.Term{Real}) = NiLang.AD.GVar(x, zero(x))
NiLang.AD.GVar(x::SymbolicUtils.Sym{Real}) = NiLang.AD.GVar(x, zero(x))
Base.convert(::Type{SymbolicUtils.Sym{Real}}, x::Integer) = SymbolicUtils.Sym{Real}(Symbol(x))
Base.convert(::Type{SymbolicUtils.Term{Real}}, x::Integer) = SymbolicUtils.Sym{Real}(Symbol(x))

Base.zero(::Type{<:SymbolicUtils.Sym{T}}) where T = SymbolicUtils.Sym{T}(Symbol(0))
Base.zero(::Type{<:SymbolicUtils.Term{T}}) where T = SymbolicUtils.Term{T}(Symbol(0))
Base.adjoint(x::SymbolicUtils.Sym{<:Real}) = x

LinearAlgebra.dot(a::SymbolicUtils.Sym{<:Real}, b::SymbolicUtils.Sym{<:Real}) = a * b

include("sparse.jl")

using BenchmarkTools, Random
using LinearAlgebra
syms = @syms a::Real b::Real c::Real d::Real e::Real f::Real g::Real
Base.rand(r::Random.AbstractRNG, ::Type{SymbolicUtils.Sym{Real}}, i::Integer) = rand(r, syms, i)
a = sprand(SymbolicUtils.Sym{Real}, 100, 100, 0.05);
b = sprand(SymbolicUtils.Sym{Real}, 100, 100, 0.05);
@benchmark SparseArrays.dot($a, $b)
@benchmark idot(SymbolicUtils.Sym{Real}(Symbol(0)), $a, $b)
@benchmark Grad(idot)(Val(1), SymbolicUtils.Sym{Real}(Symbol(0)), $a, $b)
GVar(a)

include("Symbolics/symlib.jl")
syms = @vars a b c d e f g
Base.rand(r::Random.AbstractRNG, ::Type{<:Basic}, i::Integer) = rand(r, syms, i)
a = sprand(Basic, 100, 100, 0.05);
b = sprand(Basic, 100, 100, 0.05);
@benchmark SparseArrays.dot($a, $b)
@benchmark idot(Basic(0), $a, $b)
@benchmark Grad(idot)(Val(1), Basic(0), $a, $b)

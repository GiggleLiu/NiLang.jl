export Dup

struct Dup{T} <: Bundle{T}
    x::T
    twin::T
end
function Dup(x::T) where T
   Dup{T}(x, copy(x))
end

@fieldview NiLangCore.value(x::Dup) = x.x
Base.isapprox(a::Dup, b::Dup; kwargs...) = isapprox(a.x, b.x; kwargs...) && isapprox(a.twin, b.twin; kwargs...)

# This is a patch for loading a data to GVar correctly.
import NiLangCore

NiLangCore.loaddata(::Type{GT}, x::T) where {T, GT<:GVar{T}} = convert(GT, x)
function NiLangCore.loaddata(t::Type{VT}, x::AbstractVector) where {T, VT<:AbstractVector{T}}
    convert.(T, x)
end

function NiLangCore.loaddata(t::VT, x::AbstractVector) where {T, VT<:AbstractVector{T}}
    convert(VT, NiLangCore.loaddata.(t, x))
end

function NiLangCore.loaddata(::Type{T}, x::XT) where {N, T<:Tuple{N}, XT<:Tuple{N}}
    ntuple(i=>NiLangCore.loaddata.(T.parameters[i], [i]), N)
end

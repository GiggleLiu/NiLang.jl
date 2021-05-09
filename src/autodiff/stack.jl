# This is a patch for loading a data to GVar correctly.
import NiLangCore

NiLangCore.loaddata(::GT, x::T) where {T, GT<:GVar{T}} = convert(GT, x)
NiLangCore.loaddata(::AbstractVector{GT}, x::AbstractVector{T}) where {T,GT<:GVar{T}} = GT.(x)

@generated function NiLangCore.loaddata(t::T, x::XT) where {T, XT}
    if isprimitivetype(T)
        :(error("load data from stack error: type $($T) and $($XT)) does not match"))
    else
        Expr(:new, T, [:($(NiLangCore.loaddata)(t.$NAME, x.$NAME)) for NAME in fieldnames(T)]...)
    end
end

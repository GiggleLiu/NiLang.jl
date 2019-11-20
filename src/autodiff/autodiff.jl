module ADLib

using ..NiLang
using NiLangCore
using Reexport
@reexport using NiLangCore.AD
using MLStyle

Base.:-(x::GVar) = GVar(-x.x, -x.g)
Base.:-(x::Loss) = Loss(-x.x)
NiLangCore.chfield(x::T, ::typeof(-), y::T) where T = -y

include("instructs.jl")
end

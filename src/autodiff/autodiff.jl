module ADLib

using ..NiLang
using NiLangCore
using Reexport
@reexport using NiLangCore.AD
using MLStyle

include("instructs.jl")

for op in [:>, :<, :>=, :<=, :isless]
    @eval Base.$op(a::Bundle, b::Bundle) = $op(val(a), val(b))
end
end

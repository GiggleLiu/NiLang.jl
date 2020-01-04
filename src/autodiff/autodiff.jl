module AD

using ..NiLang
using NiLangCore
using Reexport
@reexport using NiLangCore.ADCore
using MLStyle

import ..NiLang: ⊕, ⊖, NEG, CONJ, ROT, IROT, SWAP

NiLangCore.ADCore.GVar(x::Dup) = Dup(GVar(x.x), GVar(x.twin))
(invg::Type{Inv{GVar}})(x::Dup) = Dup(invg(x.x), invg(x.twin))

include("instructs_basic.jl")
include("instructs_ext.jl")
include("simple_hessian.jl")
include("taylor.jl")

for op in [:>, :<, :>=, :<=, :isless]
    @eval Base.$op(a::Bundle, b::Bundle) = $op(value(a), value(b))
end
#Base.conj(x::GVar) = GVar(x.x', x.g')
end

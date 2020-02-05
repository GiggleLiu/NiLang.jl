module AD

using ..NiLang
using NiLangCore
using Reexport
@reexport using NiLangCore.ADCore
using MLStyle

import ..NiLang: ⊕, ⊖, NEG, CONJ, ROT, IROT, SWAP

include("instructs_basic.jl")
include("instructs_ext.jl")
include("simple_hessian.jl")
include("taylor.jl")

end

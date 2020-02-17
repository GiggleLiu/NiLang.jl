module AD

using ..NiLang
using NiLangCore
using MLStyle, TupleTools

import ..NiLang: ⊕, ⊖, NEG, CONJ, ROT, IROT, SWAP, chfield, value

export GVar, grad, Loss, NoGrad, @nograd

include("vars.jl")
include("gradfunc.jl")
include("checks.jl")

include("instructs_basic.jl")
include("instructs_ext.jl")
include("simple_hessian.jl")
include("taylor.jl")

end

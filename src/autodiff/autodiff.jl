module AD

using ..NiLang
using NiLangCore
using MLStyle, TupleTools

import ..NiLang: ⊕, ⊖, NEG, ROT, IROT, SWAP,
    chfield, value

export GVar, grad, Loss, NoGrad, @nograd

include("vars.jl")
include("gradfunc.jl")
include("checks.jl")

include("instructs.jl")
include("jacobian.jl")
include("hessian_backback.jl")
include("taylor.jl")

end

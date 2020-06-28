module AD

using ..NiLang
using NiLangCore
using MatchCore, TupleTools

import ..NiLang: ROT, IROT, SWAP,
    chfield, value, NoGrad, loaddata, INC, DEC

export GVar, grad, Loss, NoGrad, @nograd

include("vars.jl")
include("gradfunc.jl")
include("checks.jl")

include("instructs.jl")
include("jacobian.jl")
include("hessian_backback.jl")
include("complex.jl")

end

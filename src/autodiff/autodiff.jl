module AD

using ..NiLang
using NiLangCore
using MLStyle, TupleTools

import ..NiLang: ROT, IROT, SWAP,
    chfield, value, NoGrad, INC, DEC, HADAMARD,
    AddConst, SubConst, NEG, INV
using NiLangCore: default_constructor

export GVar, grad, Loss, NoGrad, @nograd

include("vars.jl")
include("stack.jl")
include("gradfunc.jl")
include("checks.jl")

include("instructs.jl")
include("ulog.jl")
include("jacobian.jl")
include("hessian_backback.jl")
include("complex.jl")

if Base.VERSION >= v"1.4.2"
    include("precompile.jl")
    _precompile_()
end

end

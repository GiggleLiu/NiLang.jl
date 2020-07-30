module NiLang

using Reexport
@reexport using NiLangCore
import NiLangCore: ⊕, ⊖

using FixedPointNumbers: Q20f43, Fixed
export Fixed43
const Fixed43 = Q20f43

include("utils.jl")
include("vars.jl")
include("instructs.jl")
include("ulog.jl")
include("stack.jl")
include("complex.jl")
include("autobcast.jl")

include("autodiff/autodiff.jl")

include("stdlib/stdlib.jl")

include("deprecations.jl")

export AD
end # module

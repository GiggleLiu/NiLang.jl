module NiLang

using Reexport
@reexport using NiLangCore
import NiLangCore: ⊕, ⊖

using FixedPointNumbers: Q20f43, Fixed
export Fixed43
const Fixed43 = Q20f43

include("utils.jl")
include("instructs.jl")
include("complex.jl")
include("autobcast.jl")

include("autodiff/autodiff.jl")

include("deprecations.jl")

export AD
end # module

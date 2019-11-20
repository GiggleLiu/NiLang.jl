module NiLang

using Reexport
@reexport using NiLangCore
import NiLangCore: ⊕, ⊖

include("instructs.jl")
include("autodiff/autodiff.jl")

export AD
end # module

module NiLang

using Reexport
@reexport using NiLangCore
import NiLangCore: ⊕, ⊖

include("utils.jl")
include("instructs.jl")
include("functions.jl")
include("nnlib.jl")

include("autodiff/autodiff.jl")

export AD
end # module

module NiLang

using Reexport
@reexport using NiLangCore
import NiLangCore: invtype

using FixedPointNumbers: Q20f43, Fixed
import NiLangCore: empty_global_stacks!, loaddata
export Fixed43
const Fixed43 = Q20f43

include("utils.jl")
include("wrappers.jl")
include("vars.jl")
include("instructs.jl")
include("ulog.jl")
include("complex.jl")
include("autobcast.jl")
include("macros.jl")

include("autodiff/autodiff.jl")

include("stdlib/stdlib.jl")

include("deprecations.jl")

export AD

project_relative_path(xs...) = normpath(joinpath(dirname(dirname(pathof(@__MODULE__))), xs...))

if Base.VERSION >= v"1.4.2"
    include("precompile.jl")
    _precompile_()
end

end # module

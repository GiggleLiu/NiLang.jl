using NiLang, NiLang.AD
using BenchmarkTools

include("../exmamples/besselj.jl")

# To test this function, we first define input parameters `a` and output `out!`
a = 1.0
out! = 0.0

# We wrap the output with a randomly initialized gradient field, suppose we get the gradients from a virtual loss function.
# Also, we need to initialize an empty gradient field for elements in input cuda tensor `a`.
out! = ibesselj(out!, 2, a)[1]
out_g! = GVar(out!, 1.0)
a_g = GVar(a)

# Call the inverse program, the multiple dispatch will drive you to the goal.
println("Benchmarking NiLang")
display(@benchmark ibesselj($out!, 2, $a))
println("Benchmarking NiLang.AD")
display(@benchmark (~ibesselj)($out_g!, 2, $a_g))

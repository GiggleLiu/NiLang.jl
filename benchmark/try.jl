include("../NiLangCore/src/lens.jl")
using Lens

struct A
    a
    b::Float64
    c
    A() = new(0,0.0,0)
    A(x, y, z) = new(x, y, z)
end

a = A()
using BenchmarkTools
@benchmark @set $a.b = 3

@benchmark @with $a.b = 3
@benchmark chfield($a, Val(:b), 3)


@generated function chfield(x, ::Val{FIELD}, xval) where FIELD
    :(@with x.$FIELD = xval)
end

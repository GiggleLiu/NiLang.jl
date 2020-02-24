using NiLang, NiLang.AD, Zygote

using NiFunctions: ibesselj
using SpecialFunctions

res = besselj'(2, 1.0)

Zygote.@adjoint function besselj(n::Int, x::Number)
    out = besselj(n, x)
    out, δy -> (nothing, (~ibesselj)(GVar(out, δy), 2, GVar(x))[3].g)
end

@test besselj'(2, 1.0) ≈ res

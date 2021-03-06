# # [How to port NiLang to ChainRules](@id port_chainrules)
#
# In [How to port NiLang to Zygote](@ref port_zygote) we showed the way to insert Nilang-based
# gradient as Zygote's pullback/adjoint. Given that [ChainRules](https://github.com/JuliaDiff/ChainRules.jl)
# is now the core of many AD packages including Zygote, extending `ChainRules.rrule` with Nilang
# does the same job, except that it affects all ChainRules-based AD packages and not just Zygote.
#
# We'll use the same example as [How to port NiLang to Zygote](@ref port_zygote), so you might need
# to restart your Julia to get a fresh environment.

using NiLang, NiLang.AD, Zygote, ChainRules

# Let's start from the Julia native implementation of `norm2` function.
function norm2(x::AbstractArray{T}) where T
    out = zero(T)
    for i=1:length(x)
        @inbounds out += x[i]^2
    end
    return out
end

# Zygote is able to generate correct dual function, i.e., gradients, but much slower than the primal
# function `norm2`
using BenchmarkTools
x = randn(1000);
original_grad = norm2'(x)
@benchmark norm2'($x) seconds=1

# The primal function is
@benchmark norm2($x) seconds=1

# Then we have the reversible implementation
@i function r_norm2(out::T, x::AbstractArray{T}) where T
    for i=1:length(x)
        @inbounds out += x[i]^2
    end
end

# The gradient generated by NiLang is much faster, which is comparable to the forward program
@benchmark (~r_norm2)(GVar($(norm2(x)), 1.0), $(GVar(x))) seconds=1

# By defining our custom `rrule` using Nilang's gradient implementation, `Zygote` automaticallly
# gets boosted because it internally uses the available ChainRules ruleset.
# Here we need to create a new symbol here because otherwise Zygote will still use the
# previously generated slow implementation.
norm2_faster(x) = norm2(x)
function ChainRules.rrule(::typeof(norm2_faster), x::AbstractArray{T}) where T
    out = norm2_faster(x)
    function pullback(ȳ)
        ChainRules.NoTangent(), grad((~r_norm2)(GVar(out, ȳ), GVar(x))[2])
    end
    out, pullback
end
@assert norm2_faster'(x) ≈ original_grad

# See, much faster
@benchmark norm2_faster'(x) seconds=1

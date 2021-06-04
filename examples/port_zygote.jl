# # [How to port NiLang to Zygote](@id port_zygote)
#
# In this demo we'll show how to insert NiLang's gradient implementation to boost Zygote's gradient.
# A similar demo for ChainRules can be found in [How to port NiLang to ChainRules](@ref port_chainrules).

using NiLang, NiLang.AD, Zygote

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

# to enjoy the speed of `NiLang` in `Zygote`, just bind the adjoint rule
Zygote.@adjoint function norm2(x::AbstractArray{T}) where T
    out = norm2(x)
    out, δy -> (grad((~r_norm2)(GVar(out, δy), GVar(x))[2]),)
end
@assert norm2'(x) ≈ original_grad

# See, much faster
@benchmark norm2'(x) seconds=1

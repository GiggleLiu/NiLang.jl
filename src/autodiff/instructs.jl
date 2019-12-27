@i function ⊖(exp)(out!::GVar, x::GVar{T}) where T
    value(out!) -= exp(value(x))
    @anc anc1 = zero(T)
    anc1 += exp(value(x))
    grad(x) += grad(out!) * anc1
    anc1 -= exp(value(x))
end

@i function ⊖(log)(out!::GVar, x::GVar{T}) where T
    value(out!) -= log(value(x))
    grad(x) += grad(out!) / value(x)
end

@i function ⊖(sin)(out!::GVar, x::GVar{T}) where T
    value(out!) -= sin(value(x))
    @anc anc1 = zero(T)
    anc1 += cos(value(x))
    grad(x) += grad(out!) * anc1
    anc1 -= cos(value(x))
end

@i function ⊖(cos)(out!::GVar, x::GVar{T}) where T
    value(out!) -= cos(value(x))
    @anc anc1 = zero(T)
    anc1 -= sin(value(x))
    grad(x) += grad(out!) * anc1
    anc1 += sin(value(x))
end

for op in [:exp, :log, :sin, :cos]
    @eval @nograd ⊖($op)(out!, x::GVar)
    @eval @nograd ⊖($op)(out!::GVar, x)
end

@i function IROT(a!::GVar, b!::GVar, θ::GVar)
    IROT(value(a!), value(b!), value(θ))
    NEG(value(θ))
    value(θ) ⊖ π/2
    ROT(grad(a!), grad(b!), value(θ))
    grad(θ) += value(a!) * grad(a!)
    grad(θ) += value(b!) * grad(b!)
    value(θ) ⊕ π/2
    NEG(value(θ))
    ROT(grad(a!), grad(b!), π/2)
end

@i function IROT(a!::GVar, b!::GVar, θ)
    IROT(value(a!), value(b!), θ)
    NEG(θ)
    θ ⊖ π/2
    ROT(grad(a!), grad(b!), θ)
    θ ⊕ π/2
    NEG(θ)
    ROT(grad(a!), grad(b!), π/2)
end

@nograd IROT(a!, b!, θ::GVar)

#=
# ugly patch
@i function ⊖(GVar{Float64,Float64})(a!::GVar, b::T) where T
    value(a!) ⊖ value(b)
end
=#

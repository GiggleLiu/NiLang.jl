# unary
@i @inline function NEG(a!::GVar)
    NEG(value(a!))
    NEG(grad(a!))
end

# +-
@i @inline function ⊖(identity)(a!::GVar, b::GVar)
    value(a!) ⊖ value(b)
    grad(b) ⊕ grad(a!)
end
@nograd ⊖(identity)(a!::Real, b::GVar)
@nograd ⊖(identity)(a!::GVar, b::Real)

# +- (triple)
@i @inline function ⊖(+)(out!::GVar, x::GVar, y::GVar)
    value(out!) -= value(x) + value(y)
    grad(x) += identity(grad(out!))
    grad(y) += identity(grad(out!))
end

@i @inline function ⊖(+)(out!::GVar, x::GVar, y::Real)
    value(out!) -= value(x) + value(y)
    grad(x) += identity(grad(out!))
end

@i @inline function ⊖(+)(out!::GVar, x::Real, y::GVar)
    value(out!) -= value(x) + value(y)
    grad(y) += identity(grad(out!))
end

@i @inline function ⊖(-)(out!::GVar, x::GVar, y::GVar)
    value(out!) -= value(x) - value(y)
    grad(x) += identity(grad(out!))
    grad(y) -= identity(grad(out!))
end

@i @inline function ⊖(-)(out!::GVar, x::Real, y::GVar)
    value(out!) -= value(x) - value(y)
    grad(y) -= identity(grad(out!))
end

@i @inline function ⊖(-)(out!::GVar, x::GVar, y::Real)
    value(out!) -= value(x) - value(y)
    grad(x) += identity(grad(out!))
end

# NOTE: it will error on `SWAP(a!::GVar, b)` or `SWAP(a!, b:GVar)`
@i @inline function SWAP(a!::GVar, b!::GVar)
    SWAP(value(a!), value(b!))
    SWAP(grad(a!), grad(b!))
end

# */
@i @inline function ⊖(*)(out!::GVar, x::GVar, y::GVar)
    value(out!) -= value(x) * value(y)
    grad(x) += grad(out!) * value(y)
    grad(y) += value(x) * grad(out!)
end

@i @inline function ⊖(*)(out!::GVar, x::Real, y::GVar)
    value(out!) -= value(x) * value(y)
    grad(y) += value(x) * grad(out!)
end

@i @inline function ⊖(*)(out!::GVar, x::GVar, y::Real)
    value(out!) -= value(x) * value(y)
    grad(x) += grad(out!) * value(y)
end

@i @inline function ⊖(/)(out!::GVar{T}, x::GVar, y::GVar) where T
    value(out!) -= value(x)/value(y)
    @routine @invcheckoff begin
        a1 ← zero(grad(out!))
        a2 ← zero(grad(out!))
        a1 += value(x)*grad(out!)
        a2 += a1/value(y)
    end
    grad(x) += grad(out!)/value(y)
    grad(y) -= a2/value(y)
    ~@routine
end

@i @inline function ⊖(/)(out!::GVar{T}, x::Real, y::GVar) where T
    value(out!) -= x/value(y)
    @routine @invcheckoff begin
        a1 ← zero(grad(out!))
        a2 ← zero(grad(out!))
        a1 += x*grad(out!)
        a2 += a1/value(y)
    end
    grad(y) -= a2/value(y)
    ~@routine
end

@i @inline function ⊖(/)(out!::GVar, x::GVar, y::Real)
    value(out!) -= value(x)/y
    grad(x) += grad(out!)/y
end

@i @inline function ⊖(^)(out!::GVar{T}, x::GVar, n::GVar) where T
    ⊖(^)(value(out!), value(x), value(n))

    # grad x
    @routine @invcheckoff begin
        anc1 ← zero(value(x))
        anc2 ← zero(value(x))
        anc3 ← zero(value(x))
        jac1 ← zero(value(x))
        jac2 ← zero(value(x))

        value(n) -= identity(1)
        anc1 += value(x)^value(n)
        value(n) += identity(1)
        jac1 += anc1 * value(n)

        # get grad of n
        anc2 += log(value(x))
        anc3 += value(x) ^ value(n)
        jac2 += anc3*anc2
    end
    grad(x) += grad(out!) * jac1
    grad(n) += grad(out!) * jac2
    ~@routine
end

@i @inline function ⊖(^)(out!::GVar{T}, x::GVar, n::Real) where T
    ⊖(^)(value(out!), value(x), n)
    @routine @invcheckoff begin
        anc1 ← zero(value(x))
        jac ← zero(value(x))

        value(n) -= identity(1)
        anc1 += value(x)^n
        value(n) += identity(1)
        jac += anc1 * n
    end
    grad(x) += grad(out!) * jac
    ~@routine
end

@i @inline function ⊖(^)(out!::GVar{T}, x::Real, n::GVar) where T
    ⊖(^)(value(out!), x, value(n))
    # get jac of n
    @routine @invcheckoff begin
        anc1 ← zero(x)
        anc2 ← zero(x)
        jac ← zero(x)

        anc1 += log(x)
        anc2 += x ^ value(n)
        jac += anc1*anc2
    end
    grad(n) += grad(out!) * jac
    ~@routine
end

@i @inline function ⊖(atan)(out!::GVar{T}, y::GVar, x::GVar) where T
    ⊖(atan)(value(out!), value(y), value(x))
    @routine @invcheckoff begin
        xy2 ← zero(T)
        jac_x ← zero(T)
        jac_y ← zero(T)
        xy2 += abs2(value(x))
        xy2 += abs2(value(y))
        jac_y += value(x) / xy2
        jac_x += (-value(y)) / xy2
    end
    grad(y) += grad(out!) * jac_y
    grad(x) += grad(out!) * jac_x
    ~@routine
end

@i @inline function ⊖(atan)(out!::GVar{T}, x::GVar) where T
    ⊖(atan)(value(out!), value(x))
    @routine @invcheckoff begin
        xy2 ← zero(T)
        xy2 += abs2(value(x))
        xy2 += identity(1)
    end
    grad(x) += grad(out!) / xy2
    ~@routine
end

@i @inline function ⊖(abs)(out!::GVar, x::GVar{T}) where T
    value(out!) -= abs(value(x))
    if (x > 0, ~)
        grad(x) ⊕ grad(out!)
    else
        grad(x) ⊖ grad(out!)
    end
end

@i @inline function ⊖(abs2)(out!::GVar, x::GVar{T}) where T
    value(out!) -= abs2(value(x))
    @routine @invcheckoff begin
        x2 ← zero(T)
        x2 += 2 * value(x)
    end
    grad(x) += grad(out!) * x2
    ~@routine
end
@nograd ⊖(abs2)(a!::GVar, b::Real)
@nograd ⊖(abs2)(a!::Real, b::GVar)

for op in [:*, :/, :^, :+, :-]
    @eval @nograd ⊖($op)(out!::GVar, x::Real, y::Real)
    @eval @nograd ⊖($op)(out!::Real, x::Real, y::GVar)
    @eval @nograd ⊖($op)(out!::Real, x::GVar, y::GVar)
    @eval @nograd ⊖($op)(out!::Real, x::GVar, y::Real)
end

@i @inline function ⊖(exp)(out!::GVar, x::GVar{T}) where T
    @routine @invcheckoff begin
        anc1 ← zero(value(x))
        anc1 += exp(value(x))
    end
    value(out!) -= identity(anc1)
    grad(x) += grad(out!) * anc1
    ~@routine
end

@i @inline function ⊖(log)(out!::GVar, x::GVar{T}) where T
    value(out!) -= log(value(x))
    grad(x) += grad(out!) / value(x)
end

@i @inline function ⊖(sin)(out!::GVar, x::GVar{T}) where T
    value(out!) -= sin(value(x))
    @routine @invcheckoff begin
        anc1 ← zero(value(x))
        anc1 += cos(value(x))
    end
    grad(x) += grad(out!) * anc1
    ~@routine
end

@i @inline function ⊖(cos)(out!::GVar, x::GVar{T}) where T
    value(out!) -= cos(value(x))
    @routine @invcheckoff begin
        anc1 ← zero(value(x))
        anc1 -= sin(value(x))
    end
    grad(x) += grad(out!) * anc1
    ~@routine
end

@i @inline function ⊖(sincos)(out!::Tuple{T1,T1}, x::GVar{T}) where {T1<:GVar, T}
    @routine @invcheckoff begin
        s ← zero(T)
        c ← zero(T)
        (s, c) += sincos(value(x))
    end
    value.(out!) -= identity((s, c))
    grad(x) += grad(tget(out!, 1)) * c
    grad(x) -= grad(tget(out!, 2)) * s
    ~@routine
end

for op in [:exp, :log, :sin, :cos]
    @eval @nograd ⊖($op)(out!::Real, x::GVar)
    @eval @nograd ⊖($op)(out!::GVar, x::Real)
end

@nograd ⊖(sincos)(out!::Tuple{<:Real,<:Real}, x::GVar)
@nograd ⊖(sincos)(out!::Tuple{<:GVar,<:GVar}, x::Real)

@i @inline function IROT(a!::GVar, b!::GVar, θ::GVar)
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

@i @inline function IROT(a!::GVar, b!::GVar, θ::Real)
    IROT(value(a!), value(b!), θ)
    NEG(θ)
    θ ⊖ π/2
    ROT(grad(a!), grad(b!), θ)
    θ ⊕ π/2
    NEG(θ)
    ROT(grad(a!), grad(b!), π/2)
end

@nograd IROT(a!::Real, b!::Real, θ::GVar)

export primitive_grad
function primitive_grad end

@i function (mf::MinusEq)(out!::GVar, args...; kwargs...)
    value(out!) -= mf.f(value.(args)...; kwargs...)
    grad.(args) .+= (@skip! grad(out!)) .* primitive_grad(mf.f, value.(args)...; kwargs...)
end

@i function (mf::MinusEq)(out!::GVar, x::GVar; kwargs...)
    value(out!) -= mf.f(value(x); kwargs...)
    grad(x) += identity(primitive_grad(mf.f, value(x); kwargs...))
end

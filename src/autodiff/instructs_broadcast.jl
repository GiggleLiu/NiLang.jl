const VecGVar{T} = GVar{T, <:AbstractVector}

# unary
@i @inline function NEG(a!::VecGVar)
    NEG(value(a!))
    for i=1:length(grad(a!))
        NEG(grad(a!)[i])
    end
end

@i @inline function CONJ(a!::VecGVar)
    CONJ(value(a!))
    for i=1:length(grad(a!))
        CONJ(grad(a!)[i])
    end
end

# +-
@i @inline function ⊖(identity)(a!::VecGVar, b::VecGVar)
    value(a!) ⊖ value(b)
    for i=1:length(grad(a!))
        grad(b)[i] += identity(grad(a!)[i])
    end
end
@nograd ⊖(identity)(a!, b::VecGVar)
@nograd ⊖(identity)(a!::VecGVar, b)

# NOTE: it will error on `SWAP(a!::VecGVar, b)` or `SWAP(a!, b:VecGVar)`
@i @inline function SWAP(a!::VecGVar, b!::VecGVar)
    SWAP(value(a!), value(b!))
    for i=1:length(grad(a!))
        SWAP(grad(a!)[i], grad(b!)[i])
    end
end

# */
@i @inline function ⊖(*)(out!::VecGVar, x::VecGVar, y::VecGVar)
    value(out!) -= value(x) * value(y)
    for i=1:length(grad(out!))
        grad(x)[i] += grad(out!)[i] * value(y)
        grad(y)[i] += value(x) * grad(out!)[i]
    end
end

@i @inline function ⊖(*)(out!::VecGVar, x, y::VecGVar)
    value(out!) -= value(x) * value(y)
    for i=1:length(grad(out!))
        grad(y)[i] += value(x) * grad(out!)[i]
    end
end

@i @inline function ⊖(*)(out!::VecGVar, x::VecGVar, y)
    value(out!) -= value(x) * value(y)
    for i=1:length(grad(out!))
        grad(x)[i] += grad(out!)[i] * value(y)
    end
end

@i @inline function ⊖(/)(out!::VecGVar{T}, x::VecGVar, y::VecGVar) where T
    value(out!) -= value(x)/value(y)
    a1 ← zero(T)
    a2 ← zero(T)
    for i=1:length(grad(out!))
        grad(x)[i] += grad(out!)[i]/value(y)
        a1 += value(x)*grad(out!)[i]
        a2 += a1/value(y)
        grad(y)[i] -= a2/value(y)
        a2 -= a1/value(y)
        a1 -= value(x)*grad(out!)[i]
    end
end

@i @inline function ⊖(/)(out!::VecGVar{T}, x, y::VecGVar) where T
    value(out!) -= x/value(y)
    a1 ← zero(T)
    a2 ← zero(T)
    for i=1:length(grad(out!))
        a1 += x*grad(out!)[i]
        a2 += a1/value(y)
        grad(y)[i] -= a2/value(y)
        a2 -= a1/value(y)
        a1 -= x*grad(out!)[i]
    end
end

@i @inline function ⊖(/)(out!::VecGVar, x::VecGVar, y)
    value(out!) -= value(x)/y
    for i=1:length(grad(out!))
        grad(x)[i] += grad(out!)[i]/y
    end
end

@i @inline function ⊖(^)(out!::VecGVar{T}, x::VecGVar, n::VecGVar) where T
    ⊖(^)(value(out!), value(x), value(n))
    anc1 ← zero(T)
    anc2 ← zero(T)
    jac ← zero(T)

    # grad x
    @routine begin
        n ⊖ 1
        anc1 += value(x)^value(n)
        n ⊕ 1
        jac += anc1 * value(n)
    end
    for i=1:length(grad(out!))
        grad(x)[i] += grad(out!)[i] * jac
    end
    ~@routine

    # get grad of n
    @routine begin
        anc1 += log(value(x))
        anc2 += value(x) ^ value(n)
        jac += anc1*anc2
    end
    for i=1:length(grad(out!))
        grad(n)[i] += grad(out!)[i] * jac
    end
    ~@routine
end

@i @inline function ⊖(^)(out!::VecGVar{T}, x::VecGVar, n) where T
    ⊖(^)(value(out!), value(x), n)
    anc1 ← zero(T)
    anc2 ← zero(T)
    jac ← zero(T)

    @routine begin
        n ⊖ 1
        anc1 += value(x)^n
        n ⊕ 1
        jac += anc1 * n
    end
    for i=1:length(grad(out!))
        grad(x)[i] += grad(out!)[i] * jac
    end
    ~@routine
end

@i @inline function ⊖(^)(out!::VecGVar{T}, x, n::VecGVar) where T
    ⊖(^)(value(out!), x, value(n))
    anc1 ← zero(T)
    anc2 ← zero(T)
    jac ← zero(T)

    # get jac of n
    @routine begin
        anc1 += log(x)
        anc2 += x ^ value(n)
        jac += anc1*anc2
    end
    for i=1:length(grad(out!))
        grad(n)[i] += grad(out!)[i] * jac
    end
    ~@routine
end

@i @inline function ⊖(abs)(out!::VecGVar, x::VecGVar{T}) where T
    value(out!) -= abs(value(x))
    if (unwrap(x) > 0, ~)
        for i=1:length(grad(out!))
            grad(x)[i] += identity(grad(out!)[i])
        end
    else
        for i=1:length(grad(out!))
            grad(x)[i] -= identity(grad(out!)[i])
        end
    end
end

@i @inline function ⊖(exp)(out!::VecGVar, x::VecGVar{T}) where T
    value(out!) -= exp(value(x))
    anc1 ← zero(T)
    anc1 += exp(value(x))
    for i=1:length(grad(out!))
        grad(x)[i] += grad(out!)[i] * anc1
    end
    anc1 -= exp(value(x))
end

@i @inline function ⊖(log)(out!::VecGVar, x::VecGVar{T}) where T
    value(out!) -= log(value(x))
    for i=1:length(grad(out!))
        grad(x)[i] += grad(out!)[i] / value(x)
    end
end

@i @inline function ⊖(sin)(out!::VecGVar, x::VecGVar{T}) where T
    value(out!) -= sin(value(x))
    anc1 ← zero(T)
    anc1 += cos(value(x))
    for i=1:length(grad(out!))
        grad(x)[i] += grad(out!)[i] * anc1
    end
    anc1 -= cos(value(x))
end

@i @inline function ⊖(cos)(out!::VecGVar, x::VecGVar{T}) where T
    value(out!) -= cos(value(x))
    anc1 ← zero(T)
    anc1 -= sin(value(x))
    for i=1:length(grad(out!))
        grad(x)[i] += grad(out!)[i] * anc1
    end
    anc1 += sin(value(x))
end

for op in [:exp, :log, :sin, :cos]
    @eval @nograd ⊖($op)(out!, x::VecGVar)
    @eval @nograd ⊖($op)(out!::VecGVar, x)
end

@i @inline function IROT(a!::VecGVar, b!::VecGVar, θ::VecGVar)
    IROT(value(a!), value(b!), value(θ))
    NEG(value(θ))
    value(θ) ⊖ π/2
    for i=1:length(grad(a!))
        ROT(grad(a!)[i], grad(b!)[i], value(θ))
        grad(θ)[i] += value(a!) * grad(a!)[i]
        grad(θ)[i] += value(b!) * grad(b!)[i]
    end
    value(θ) ⊕ π/2
    NEG(value(θ))
    for i=1:length(grad(a!))
        ROT(grad(a!)[i], grad(b!)[i], π/2)
    end
end

@i @inline function IROT(a!::VecGVar, b!::VecGVar, θ)
    IROT(value(a!), value(b!), θ)
    NEG(θ)
    θ ⊖ π/2
    for i=1:length(grad(a!))
        ROT(grad(a!)[i], grad(b!)[i], θ)
    end
    θ ⊕ π/2
    NEG(θ)
    for i=1:length(grad(a!))
        ROT(grad(a!)[i], grad(b!)[i], π/2)
    end
end

@nograd IROT(a!, b!, θ::VecGVar)

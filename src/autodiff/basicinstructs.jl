# unary
@i function NEG(a!::GVar)
    NEG(value(a!))
    NEG(grad(a!))
end

@i function CONJ(a!::GVar)
    CONJ(value(a!))
    CONJ(grad(a!))
end

@i function SWAP(a!::GVar, b!::GVar)
    SWAP(value(b!), value(a!))
    SWAP(grad(b!), grad(a!))
end

# +-
@i function ⊖(identity)(a!::GVar, b::GVar)
    value(a!) ⊖ value(b)
    grad(b) ⊕ grad(a!)
end

# */
@i function ⊖(*)(out!::GVar, x::GVar, y::GVar)
    value(out!) -= value(x) * value(y)
    grad(x) += grad(out!) * value(y)'
    grad(y) += value(x)' * grad(out!)
end

@i function ⊖(*)(out!::GVar, x, y::GVar)
    value(out!) -= value(x) * value(y)
    grad(y) += value(x)' * grad(out!)
end

@i function ⊖(*)(out!::GVar, x, y::GVar)
    value(out!) -= value(x) * value(y)
    grad(x) += grad(out!) * value(y)'
end

@i function ⊖(/)(out!::GVar{T}, x::GVar, y::GVar) where T
    value(out!) -= value(x)/value(y)
    grad(x) += grad(out!)/value(y)
    @anc a1 = zero(T)
    @anc a2 = zero(T)
    a1 += value(x)*grad(out!)
    a2 += value(a1)/value(y)
    grad(y) -= a2/value(y)
    a2 -= value(a1)/value(y)
    a1 -= value(x)*grad(out!)
end

@i function ⊖(/)(out!::GVar{T}, x, y::GVar) where T
    value(out!) -= value(x)/value(y)
    @anc a1 = zero(T)
    @anc a2 = zero(T)
    a1 += value(x)*grad(out!)
    a2 += value(a1)/value(y)
    grad(y) -= a2/value(y)
    a2 -= value(a1)/value(y)
    a1 -= value(x)*grad(out!)
end

@i function ⊖(/)(out!::GVar, x::GVar, y)
    value(out!) -= value(x)/value(y)
    grad(x) += grad(out!)/value(y)
end

@i function ⊖(^)(out!::GVar{T}, x, n) where T
    ⊖(^)(value(out!), value(x), value(n))
    @anc anc1 = zero(T)
    @anc anc2 = zero(T)
    @anc jac = zero(T)
    if (x isa GVar && value(n) != 0, ~)
        @routine getjac begin
            n ⊖ 1
            anc1 += value(x)^n
            n ⊕ 1
            jac += anc1 * n
        end
        grad(x) += grad(out!) * jac
        ~@routine getjac
    end

    if (n isa GVar, ~)
        # get jac of n
        @routine getnjac begin
            anc1 += log(value(x))
            anc2 += value(x) ^ value(n)
            jac += anc1*anc2
        end
        grad(n) += grad(out!) * jac
        ~@routine getnjac
    end
end

NiLangCore.chfield(x::Complex, ::typeof(real), r) = chfield(x, Val{:re}(), r)
NiLangCore.chfield(x::Complex, ::typeof(imag), r) = chfield(x, Val{:im}(), r)

#@i function Base.complex(y::Complex{T}, a::T, b::T) where T
#    SWAP(y.re, a)
#    SWAP(y.im, b)
#end

@i function NEG(y!::Complex{T}) where T
    NEG(y!.re)
    NEG(y!.im)
end

@i function Base.conj(y!::Complex{T}) where T
    NEG(y!.im)
end

@i function ⊕(angle)(r!::T, x::Complex{T}) where T
    r! += atan(x.im, x.re)
end

@i function ⊕(identity)(y!::Complex{T}, a::Complex{T}) where T
    y!.re += identity(a.re)
    y!.im += identity(a.im)
end

@inline function SWAP(a!::Complex, b!::Complex)
    b!, a!
end

@i function ⊕(abs2)(y!::T, a::Complex{T}) where T
    y! += a.re^2
    y! += a.im^2
end

@i function ⊕(abs)(y!::T, a::Complex{T}) where T
    @routine @invcheckoff begin
        y2 ← zero(y!)
        y2 += abs2(a)
    end
    y! += y2 ^ 0.5
    ~@routine
end

@i function ⊕(*)(y!::Complex{T}, a::Complex{T}, b::Complex{T}) where T
    y!.re += a.re * b.re
    y!.re += a.im * (-b.im)
    y!.im += a.re * b.im
    y!.im += a.im * b.re
end

@i function ⊕(*)(y!::Complex{T}, a::Real, b::Complex{T}) where T
    y!.re += a * b.re
    y!.im += a * b.im
end

@i function ⊕(*)(y!::Complex{T}, a::Complex{T}, b::Real) where T
    y!.re += a.re * b
    y!.im += a.im * b
end

for OP in [:+, :-]
    @eval @i function ⊕($OP)(y!::Complex{T}, a::Complex{T}, b::Complex{T}) where T
        y!.re += $OP(a.re, b.re)
        y!.im += $OP(a.im, b.im)
    end

    @eval @i function ⊕($OP)(y!::Complex{T}, a::Complex{T}, b::Real) where T
        y!.re += $OP(a.re, b)
    end

    @eval @i function ⊕($OP)(y!::Complex{T}, a::Real, b::Complex{T}) where T
        y!.re += $OP(a, b.re)
    end
end

@i function ⊕(/)(y!::Complex{T}, a::Complex{T}, b::Complex{T}) where T
    @routine @invcheckoff begin
        b2 ← zero(T)
        ab ← zero(y!)
        b2 += abs2(b)
        conj(b)
        ab += a * b
    end
    y! += ab / b2
    ~@routine
end

@i function ⊕(/)(y!::Complex{T}, a::Complex{T}, b::Real) where T
    y!.re += a.re / b
    y!.im += a.im / b
end

@i function ⊕(/)(y!::Complex{T}, a::Real, b::Complex{T}) where T
    @routine @invcheckoff begin
        b2 ← zero(T)
        ab ← zero(y!)
        b2 += abs2(b)
        conj(b)
        ab += a * b
    end
    y! += ab / b2
    ~@routine
end

@i function ⊕(exp)(y!::Complex{T}, x::Complex{T}) where T
    @routine @invcheckoff begin
        s ← zero(T)
        c ← zero(T)
        expn ← zero(T)
        z ← zero(y!)
        (s, c) += sincos(x.im)
        SWAP(z.re, c)
        SWAP(z.im, s)
        expn += exp(x.re)
    end
    y! += expn * z
    ~@routine
end

@i function ⊕(log)(y!::Complex{T}, x::Complex{T}) where T
    @routine @invcheckoff begin
        n ← zero(T)
        n += abs(x)
    end
    y!.re += log(n)
    y!.im += angle(x)
    ~@routine
end

@i function ⊕(^)(y!::Complex{T}, a::Complex{T}, b::Real) where T
    @routine @invcheckoff begin
        r ← zero(T)
        θ ← zero(T)
        s ← zero(T)
        c ← zero(T)
        absy ← zero(T)
        bθ ← zero(T)
        r += abs(a)
        θ += angle(a)
        bθ += θ * b
        (s, c) += sincos(bθ)
        absy += r ^ b
    end
    y!.re += absy * c
    y!.im += absy * s
    ~@routine
end

for OP in [:*, :/, :+, :-, :^]
    @eval @i function ⊕($OP)(y!::Complex{T}, a::Real, b::Real) where T
        y!.re += $OP(a, b)
    end
end

for OP in [:identity, :cos, :sin, :log, :exp]
    @eval @i function ⊕($OP)(y!::Complex{T}, a::Real) where T
        y!.re += $OP(a)
    end
end

NiLangCore.chfield(x::Complex, ::typeof(real), r) = chfield(x, Val{:re}(), r)
NiLangCore.chfield(x::Complex, ::typeof(imag), r) = chfield(x, Val{:im}(), r)

@i @inline function Base.:-(y!::Complex{T}) where T
    -(y!.re)
    -(y!.im)
end

@i @inline function Base.conj(y!::Complex{T}) where T
    -(y!.im)
end

@i @inline function ⊕(angle)(r!::T, x::Complex{T}) where T
    r! += atan(x.im, x.re)
end

@i @inline function ⊕(identity)(y!::Complex{T}, a::Complex{T}) where T
    y!.re += a.re
    y!.im += a.im
end

@inline function SWAP(a!::Complex, b!::Complex)
    b!, a!
end

@i @inline function ⊕(abs2)(y!::T, a::Complex{T}) where T
    y! += a.re^2
    y! += a.im^2
end

@i @inline function ⊕(abs)(y!::T, a::Complex{T}) where T
    @routine @invcheckoff begin
        y2 ← zero(y!)
        y2 += abs2(a)
    end
    y! += y2 ^ 0.5
    ~@routine
end

@i @inline function ⊕(*)(y!::Complex{T}, a::Complex{T}, b::Complex{T}) where T
    y!.re += a.re * b.re
    y!.re += a.im * (-b.im)
    y!.im += a.re * b.im
    y!.im += a.im * b.re
end

@i @inline function ⊕(*)(y!::Complex{T}, a::Real, b::Complex{T}) where T
    y!.re += a * b.re
    y!.im += a * b.im
end

@i @inline function ⊕(*)(y!::Complex{T}, a::Complex{T}, b::Real) where T
    y!.re += a.re * b
    y!.im += a.im * b
end

for OP in [:+, :-]
    @eval @i @inline function ⊕($OP)(y!::Complex{T}, a::Complex{T}, b::Complex{T}) where T
        y!.re += $OP(a.re, b.re)
        y!.im += $OP(a.im, b.im)
    end

    @eval @i @inline function ⊕($OP)(y!::Complex{T}, a::Complex{T}, b::Real) where T
        y!.re += $OP(a.re, b)
    end

    @eval @i @inline function ⊕($OP)(y!::Complex{T}, a::Real, b::Complex{T}) where T
        y!.re += $OP(a, b.re)
    end
end

@i @inline function ⊕(/)(y!::Complex{T}, a::Complex{T}, b::Complex{T}) where T
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

@i @inline function ⊕(/)(y!::Complex{T}, a::Complex{T}, b::Real) where T
    y!.re += a.re / b
    y!.im += a.im / b
end

@i @inline function ⊕(/)(y!::Complex{T}, a::Real, b::Complex{T}) where T
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

@i @inline function ⊕(exp)(y!::Complex{T}, x::Complex{T}) where T
    @routine @invcheckoff begin
        @zeros T s c expn
        z ← zero(y!)
        (s, c) += sincos(x.im)
        SWAP(z.re, c)
        SWAP(z.im, s)
        expn += exp(x.re)
    end
    y! += expn * z
    ~@routine
end

@i @inline function ⊕(log)(y!::Complex{T}, x::Complex{T}) where T
    @routine @invcheckoff begin
        n ← zero(T)
        n += abs(x)
    end
    y!.re += log(n)
    y!.im += angle(x)
    ~@routine
end

@i @inline function ⊕(^)(y!::Complex{T}, a::Complex{T}, b::Real) where T
    @routine @invcheckoff begin
        @zeros T r θ s c absy bθ
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

@i @inline function ⊕(complex)(y!::Complex{T}, a::T, b::T) where T
    y!.re += a
    y!.im += b
end

for OP in [:*, :/, :+, :-, :^]
    @eval @i @inline function ⊕($OP)(y!::Complex{T}, a::Real, b::Real) where T
        y!.re += $OP(a, b)
    end
end

for OP in [:identity, :cos, :sin, :log, :exp]
    @eval @i @inline function ⊕($OP)(y!::Complex{T}, a::Real) where T
        y!.re += $OP(a)
    end
end

export CONJ
NiLangCore.chfield(x::Complex, ::typeof(real), r) = chfield(x, Val{:re}(), r)
NiLangCore.chfield(x::Complex, ::typeof(imag), r) = chfield(x, Val{:im}(), r)

@i @inline function NEG(y!::Complex)
    NEG(y!.re)
    NEG(y!.im)
end

@i @inline function CONJ(y!::Complex{T}) where T
    -(y!.im)
end

@i @inline function ⊕(angle)(r!::Real, x::Complex)
    r! += atan(x.im, x.re)
end

@i @inline function ⊕(identity)(y!::Complex, a::Complex)
    y!.re += a.re
    y!.im += a.im
end

@inline function SWAP(a!::Complex, b!::Complex)
    b!, a!
end

@i @inline function ⊕(abs2)(y!::Real, a::Complex)
    y! += a.re^2
    y! += a.im^2
end

@i @inline function ⊕(abs)(y!::Real, a::Complex)
    @routine @invcheckoff begin
        y2 ← zero(y!)
        y2 += abs2(a)
    end
    y! += sqrt(y2)
    ~@routine
end

@i @inline function ⊕(*)(y!::Complex, a::Complex, b::Complex)
    y!.re += a.re * b.re
    y!.re += a.im * (-b.im)
    y!.im += a.re * b.im
    y!.im += a.im * b.re
end

@i @inline function ⊕(*)(y!::Complex, a::Real, b::Complex)
    y!.re += a * b.re
    y!.im += a * b.im
end

@i @inline function ⊕(*)(y!::Complex, a::Complex, b::Real)
    y!.re += a.re * b
    y!.im += a.im * b
end

for OP in [:+, :-]
    @eval @i @inline function ⊕($OP)(y!::Complex, a::Complex, b::Complex)
        y!.re += $OP(a.re, b.re)
        y!.im += $OP(a.im, b.im)
    end

    @eval @i @inline function ⊕($OP)(y!::Complex, a::Complex, b::Real)
        y!.re += $OP(a.re, b)
    end

    @eval @i @inline function ⊕($OP)(y!::Complex, a::Real, b::Complex)
        y!.re += $OP(a, b.re)
    end
end

@i @inline function ⊕(/)(y!::Complex, a::Complex, b::Complex{T}) where T
    @routine @invcheckoff begin
        b2 ← zero(T)
        ab ← zero(y!)
        b2 += abs2(b)
        CONJ(b)
        ab += a * b
    end
    y! += ab / b2
    ~@routine
end

@i @inline function ⊕(/)(y!::Complex, a::Complex, b::Real)
    y!.re += a.re / b
    y!.im += a.im / b
end

@i @inline function ⊕(/)(y!::Complex, a::Real, b::Complex{T}) where T
    @routine @invcheckoff begin
        b2 ← zero(T)
        ab ← zero(y!)
        b2 += abs2(b)
        CONJ(b)
        ab += a * b
    end
    y! += ab / b2
    ~@routine
end

@i @inline function :(+=)(inv)(y!::Complex, b::Complex{T}) where T
    @routine @invcheckoff begin
        b2 ← zero(real(T))
        b2 += abs2(b)
    end
    y! += b' / b2
    ~@routine
end

@i @inline function ⊕(exp)(y!::Complex, x::Complex{T}) where T
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

@i @inline function ⊕(log)(y!::Complex, x::Complex{T}) where T
    @routine @invcheckoff begin
        n ← zero(T)
        n += abs(x)
    end
    y!.re += log(n)
    y!.im += angle(x)
    ~@routine
end

@i @inline function ⊕(^)(y!::Complex, a::Complex{T}, b::Real) where T
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

@i @inline function ⊕(complex)(y!::Complex, a::Real, b::Real)
    y!.re += a
    y!.im += b
end

for OP in [:*, :/, :+, :-, :^]
    @eval @i @inline function ⊕($OP)(y!::Complex, a::Real, b::Real)
        y!.re += $OP(a, b)
    end
end

for OP in [:identity, :cos, :sin, :log, :exp]
    @eval @i @inline function ⊕($OP)(y!::Complex, a::Real)
        y!.re += $OP(a)
    end
end

@i @inline function HADAMARD(x::Complex, y::Complex)
    HADAMARD(x.re, y.re)
    HADAMARD(x.im, y.im)
end

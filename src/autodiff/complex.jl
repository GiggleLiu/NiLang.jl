@i @inline function :(+=)(angle)(r!::T, x::Complex{T}) where T<:GVar
    r! += atan(x.im, x.re)
end

@i @inline function :(+=)(abs2)(y!::T, a::Complex{T}) where T<:GVar
    y! += a.re^2
    y! += a.im^2
end

@i @inline function :(+=)(abs)(y!::T, a::Complex{T}) where T<:GVar
    @routine @invcheckoff begin
        y2 ← zero(y!)
        y2 += abs2(a)
    end
    y! += sqrt(y2)
    ~@routine
end

Base.zero(x::Complex{T}) where T<:GVar = Complex(zero(T), zero(T))
Base.zero(::Type{Complex{T}}) where T<:GVar = Complex(zero(T), zero(T))
Base.one(x::Complex{T}) where T<:GVar = Complex(one(T), zero(T))
Base.one(::Type{Complex{T}}) where T<:GVar = Complex(one(T), zero(T))

export AutoBcast

"""
    AutoBcast{T,N} <: IWrapper{T}

A vectorized variable.
"""
struct AutoBcast{T,N} <: IWrapper{T} x::Vector{T} end
AutoBcast(x::Vector{T}) where {T} = AutoBcast{T, length(x)}(x)
AutoBcast(x::AutoBcast{T,N}) where {T,N} = x # to avoid ambiguity error
AutoBcast{T,N}(x::AutoBcast{T,N}) where {T,N} = x
NiLangCore.value(x::AutoBcast) = x.x
NiLangCore.chfield(x::AutoBcast, ::typeof(value), xval) = chfield(x, Val(:x), xval)
Base.zero(x::AutoBcast) = AutoBcast(zero(x.x))
Base.zero(::Type{AutoBcast{T,N}}) where {T,N} = AutoBcast{T,N}(zeros(T, N))
Base.length(ab::AutoBcast{T,N}) where {T, N} = N

for F1 in [:(Base.:-), :INC, :FLIP, :DEC]
    @eval function $F1(a!::AutoBcast)
        @instr @invcheckoff @inbounds for i=1:length(a!)
            $F1(a!.x[i])
        end
        a!
    end
end

for F2 in [:SWAP, :((inf::PlusEq)), :((inf::MinusEq)), :((inf::XorEq))]
    F2 != :SWAP && @eval function $F2(a::AutoBcast, b::Real)
        @instr @invcheckoff @inbounds for i=1:length(a)
            $F2(a.x[i], b)
        end
        a, b
    end
    @eval function $F2(a::AutoBcast, b::AutoBcast)
        @instr @invcheckoff @inbounds for i=1:length(a)
            $F2(a.x[i], b.x[i])
        end
        a, b
    end
end

for F3 in [:ROT, :IROT, :((inf::PlusEq)), :((inf::MinusEq)), :((inf::XorEq))]
    if !(F3 in [:ROT, :IROT])
        @eval function $F3(a::AutoBcast, b::Real, c::Real)
            @instr @invcheckoff @inbounds for i=1:length(a)
                $F3(a.x[i], b, c)
            end
            a, b, c
        end
        @eval function $F3(a::AutoBcast, b::Real, c::AutoBcast)
            @instr @invcheckoff for i=1:length(a)
                $F3(a.x[i], b, c.x[i])
            end
            a, b, c
        end
    end
    @eval function $F3(a::AutoBcast, b::AutoBcast, c::Real)
        @instr @invcheckoff @inbounds for i=1:length(a)
            $F3(a.x[i], b.x[i], c)
        end
        a, b, c
    end
    @eval function $F3(a::AutoBcast, b::AutoBcast, c::AutoBcast)
        @instr @invcheckoff @inbounds for i=1:length(a)
            $F3(a.x[i], b.x[i], c.x[i])
        end
        a, b, c
    end
end

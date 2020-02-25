export AutoBcast

"""
    AutoBcast{T} <: IWrapper{T}

A vectorized variable.
"""
@pure_wrapper AutoBcast

Base.length(ab::AutoBcast) = length(ab.x)

for F1 in [:NEG, :CONJ]
    @eval @inline function $F1(a!::AutoBcast)
        @instr @invcheckoff for i=1:length(a!)
            $F1(a!.x[i])
        end
        a!
    end
end

for F2 in [:XOR, :SWAP, :MULINT, :DIVINT, :((inf::PlusEq)), :((inf::MinusEq)), :((inf::XorEq))]
    F2 != :SWAP && @eval @inline function $F2(a::AutoBcast, b)
        @instr @invcheckoff for i=1:length(a)
            $F2(a.x[i], b)
        end
        a, b
    end
    @eval @inline function $F2(a::AutoBcast, b::AutoBcast)
        @instr @invcheckoff for i=1:length(a)
            $F2(a.x[i], b.x[i])
        end
        a, b
    end
end

for F3 in [:ROT, :IROT, :((inf::PlusEq)), :((inf::MinusEq)), :((inf::XorEq))]
    if !(F3 in [:ROT, :IROT])
        @eval @inline function $F3(a::AutoBcast, b, c)
            @instr @invcheckoff for i=1:length(a)
                $F3(a.x[i], b, c)
            end
            a, b, c
        end
        @eval @inline function $F3(a::AutoBcast, b, c::AutoBcast)
            @instr @invcheckoff for i=1:length(a)
                $F3(a.x[i], b, c.x[i])
            end
            a, b, c
        end
    end
    @eval @inline function $F3(a::AutoBcast, b::AutoBcast, c)
        @instr @invcheckoff for i=1:length(a)
            $F3(a.x[i], b.x[i], c)
        end
        a, b, c
    end
    @eval @inline function $F3(a::AutoBcast, b::AutoBcast, c::AutoBcast)
        @instr @invcheckoff for i=1:length(a)
            $F3(a.x[i], b.x[i], c.x[i])
        end
        a, b, c
    end
end

export XOR, SWAP, NEG, CONJ
export ROT, IROT

function NEG(a!::Number)
    -a!
end
@selfdual NEG

function CONJ(x::Number)
    x'
end
@selfdual CONJ

function XOR(a!::Number, b::Number)
    a!⊻b, b
end
@selfdual XOR

function SWAP(a!::Number, b!::Number)
    b!, a!
end
@selfdual SWAP

function ROT(i::Number, j::Number, θ::Number)
    a, b = rot(value(i), value(j), value(θ))
    @assign value(i) a
    @assign value(j) b
    i, j, θ
end
function IROT(i::Number, j::Number, θ::Number)
    i, j, _ = ROT(i, j, -θ)
    i, j, θ
end
@dual ROT IROT

for F1 in [:NEG, :CONJ]
    @eval @i function $F1(a!)
        $F1(value(a!))
    end
end

for F2 in [:XOR, :SWAP]
    @eval @i function $F2(a, b)
        $F2(value(a), value(b))
    end
end

for F3 in [:ROT, :ITOR]
    @eval @i function $F3(a, b, c)
        $F3(value(a), value(b), value(c))
    end
end

for (TP, OP) in [(:PlusEq, :+), (:MinusEq, :-), (:XorEq, :⊻)]
    @eval @i function (inf::$TP)(out!, args...; kwargs...)
        inf(value(out!), value.(args)...; kwargs...)
    end
end

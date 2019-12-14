export XOR, SWAP, NEG, CONJ
export ROT, IROT

function XOR(a!, b)
    @assign value(a!) xor(value(a!), value(b))
    a!, b
end
@selfdual XOR

function SWAP(a!, b!)
    va = value(a!)
    @assign value(a!) value(b!)
    @assign value(b!) va
    a!, b!
end
@selfdual SWAP

function NEG(a!)
    @assign value(a!) -value(a!)
    a!
end
@selfdual NEG

function CONJ(x)
    @assign value(x) conj(value(x))
    x
end
@selfdual CONJ

function ROT(i, j, θ)
    a, b = rot(value(i), value(j), value(θ))
    @assign value(i) a
    @assign value(j) b
    i, j, θ
end
function IROT(i, j, θ)
    i, j, _ = ROT(i, j, -θ)
    i, j, θ
end
@dual ROT IROT

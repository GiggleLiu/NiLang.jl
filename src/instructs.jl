export ⊕, ⊖, XOR, SWAP, NEG, CONJ
export ROT, IROT

function ⊕(a!, b)
    @assign val(a!) val(a!) + val(b)
    a!, b
end
function ⊖(a!, b)
    @assign val(a!) val(a!) - val(b)
    a!, b
end
const _add = ⊕
const _sub = ⊖
@dual _add _sub
@ignore ⊕(a!::Nothing, b)
@ignore ⊕(a!::Nothing, b::Nothing)
@ignore ⊕(a!, b::Nothing)

function XOR(a!, b)
    @assign val(a!) xor(val(a!), val(b))
    a!, b
end
@selfdual XOR

function SWAP(a!, b!)
    va = val(a!)
    @assign val(a!) val(b!)
    @assign val(b!) va
    a!, b!
end
@selfdual SWAP

function NEG(a!)
    @assign val(a!) -val(a!)
    a!
end
@selfdual NEG

function CONJ(x)
    @assign val(x) conj(val(x))
    x
end
@selfdual CONJ

function ROT(i, j, θ)
    a, b = rot(val(i), val(j), val(θ))
    @assign val(i) a
    @assign val(j) b
    i, j, θ
end
function IROT(i, j, θ)
    i, j, _ = ROT(i, j, -θ)
    i, j, θ
end
@dual ROT IROT

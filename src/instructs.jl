export ⊕, ⊖, XOR, SWAP, NEG

@dual begin
    function ⊕(a!, b)
        @assign val(a!) val(a!) + val(b)
        a!, b
    end
    function ⊖(a!, b)
        @assign val(a!) val(a!) - val(b)
        a!, b
    end
end
@ignore ⊕(a!::Nothing, b)
@ignore ⊕(a!::Nothing, b::Nothing)
@ignore ⊕(a!, b::Nothing)

@selfdual begin
    function XOR(a!, b)
        @assign val(a!) xor(val(a!), val(b))
        a!, b
    end
end

@selfdual begin
    function SWAP(a!, b!)
        va = val(a!)
        @assign val(a!) val(b!)
        @assign val(b!) va
        a!, b!
    end
end

@selfdual begin
    function NEG(a!)
        @assign val(a!) -val(a!)
        a!
    end
end

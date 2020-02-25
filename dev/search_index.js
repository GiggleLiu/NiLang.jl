var documenterSearchIndex = {"docs":
[{"location":"#NiLang.jl-1","page":"Home","title":"NiLang.jl","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"NiLang is a reversible eDSL that can run backwards. The motation is to support source to source AD.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Our paper is comming soon.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Welcome for discussion in Julia slack, #autodiff channel.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Modules = [NiLang]","category":"page"},{"location":"#NiLang.CONJ-Tuple{Number}","page":"Home","title":"NiLang.CONJ","text":"CONJ(a!) -> a!'\n\n\n\n\n\n","category":"method"},{"location":"#NiLang.DIVINT-Tuple{Number,Integer}","page":"Home","title":"NiLang.DIVINT","text":"DIVINT(a!, b::Integer) -> a!/b, b\n\n\n\n\n\n","category":"method"},{"location":"#NiLang.IROT-Tuple{Number,Number,Number}","page":"Home","title":"NiLang.IROT","text":"IROT(a!, b!, θ) -> ROT(a!, b!, -θ)\n\n\n\n\n\n","category":"method"},{"location":"#NiLang.MULINT-Tuple{Number,Integer}","page":"Home","title":"NiLang.MULINT","text":"MULINT(a!, b::Integer) -> a!*b, b\n\n\n\n\n\n","category":"method"},{"location":"#NiLang.NEG-Tuple{Number}","page":"Home","title":"NiLang.NEG","text":"NEG(a!) -> -a!\n\n\n\n\n\n","category":"method"},{"location":"#NiLang.ROT-Tuple{Number,Number,Number}","page":"Home","title":"NiLang.ROT","text":"ROT(a!, b!, θ) -> a!', b!', θ\n\nbeginalign\n    rm ROT(a b theta)  = beginbmatrix\n        cos(theta)  - sin(theta)\n        sin(theta)   cos(theta)\n    endbmatrix\n    beginbmatrix\n        a\n        b\n    endbmatrix\nendalign\n\n\n\n\n\n","category":"method"},{"location":"#NiLang.SWAP-Tuple{Number,Number}","page":"Home","title":"NiLang.SWAP","text":"SWAP(a!, b!) -> b!, a!\n\n\n\n\n\n","category":"method"},{"location":"#NiLang.XOR-Tuple{Number,Number}","page":"Home","title":"NiLang.XOR","text":"XOR(a!, b) -> a! ⊻ b, b\n\n\n\n\n\n","category":"method"},{"location":"#NiLang.arshift-Union{Tuple{T}, Tuple{T,Any}} where T","page":"Home","title":"NiLang.arshift","text":"arshift(x, n)\n\nright shift, sign extending.\n\n\n\n\n\n","category":"method"},{"location":"#NiLang.plshift-Tuple{Any,Any}","page":"Home","title":"NiLang.plshift","text":"plshift(x, n)\n\nperiodic left shift.\n\n\n\n\n\n","category":"method"},{"location":"#NiLang.prshift-Tuple{Any,Any}","page":"Home","title":"NiLang.prshift","text":"plshift(x, n)\n\nperiodic right shift.\n\n\n\n\n\n","category":"method"},{"location":"#NiLang.rot-Tuple{Any,Any,Any}","page":"Home","title":"NiLang.rot","text":"rot(a, b, θ)\n\nrotate variables a and b by an angle θ\n\n\n\n\n\n","category":"method"}]
}

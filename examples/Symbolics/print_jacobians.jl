using NiLang, NiLang.AD

include("symlib.jl")
NiLang.AD.isvar(sym::Basic) = true

# a patch for symbolic IROT
@i @inline function NiLang.IROT(a!::GVar{<:Basic}, b!::GVar{<:Basic}, θ::GVar{<:Basic})
    IROT(value(a!), value(b!), value(θ))
    NEG(value(θ))
    value(θ) -= Basic(π)/2
    ROT(grad(a!), grad(b!), value(θ))
    grad(θ) += value(a!) * grad(a!)
    grad(θ) += value(b!) * grad(b!)
    value(θ) += Basic(π)/2
    NEG(value(θ))
    ROT(grad(a!), grad(b!), Basic(π)/2)
end

NiLang.INC(x::Basic) = x + one(x)
NiLang.DEC(x::Basic) = x - one(x)
NiLang.NEG(x::Basic) = -x
@inline function NiLang.ROT(i::Basic, j::Basic, θ::Basic)
    a, b = rot(i, j, θ)
    a, b, θ
end
@inline function NiLang.IROT(i::Basic, j::Basic, θ::Basic)
    i, j, _ = ROT(i, j, -θ)
    i, j, θ
end
Base.sincos(x::Basic) = (sin(x), cos(x))

function printall()
    syms = (Basic(:a), Basic(:b), Basic(:c))

    for subop in [identity, *, /, ^, exp, log, sin, cos]
        for opm in [⊕, ⊖]
            @show opm
            op = opm(subop)
            printone(op, syms)
        end
    end
    for op in [NEG, ROT, IROT]
        printone(op, syms)
    end
    # abs, conj
end

"""print the jacobian of one operator"""
function printone(op, syms)
    n = nargs(op)
    jac = jacobian_repeat(Basic, op, syms[1:nargs(op)])
    println("------ $op ------")
    pretty_print_matrix(jac)
end

printall()

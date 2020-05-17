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
    jac = jacobian(Basic, op, syms[1:nargs(op)])
    println("------ $op ------")
    pretty_print_matrix(jac)
end

printall()

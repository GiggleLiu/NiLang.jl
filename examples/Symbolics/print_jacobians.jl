using NiLang, NiLang.AD

include("symlib.jl")
NiLang.AD.isvar(sym::Basic) = true
NiLang.AD.GVar(sym::Basic) = GVar(sym, zero(sym))

# a patch for symbolic IROT
@i @inline function NiLang.IROT(a!::GVar{<:Basic}, b!::GVar{<:Basic}, θ::GVar{<:Basic})
    IROT(a!.x, b!.x, θ.x)
    -(θ.x)
    θ.x -= Basic(π)/2
    ROT(a!.g, b!.g, θ.x)
    θ.g += a!.x * a!.g
    θ.g += b!.x * b!.g
    θ.x += Basic(π)/2
    -(θ.x)
    ROT(a!.g, b!.g, Basic(π)/2)
end

NiLang.INC(x::Basic) = x + one(x)
NiLang.DEC(x::Basic) = x - one(x)
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
    syms = [Basic(:a), Basic(:b), Basic(:c)]

    for (subop, nargs) in [(identity, 2), (*, 3), (/, 3), (^, 3), (exp, 2), (log, 2), (sin, 2), (cos, 2)]
        for opm in [⊕, ⊖]
            op = opm(subop)
            @show op
            printone(op, syms, nargs)
        end
    end
    for (op, nargs) in [(-, 1), (ROT, 3), (IROT, 3)]
        printone(op, syms, nargs)
    end
    # abs, conj
end

@i function jf1(op, x)
    op(x[1])
end

@i function jf2(op, x)
    op(x[1], x[2])
end

@i function jf3(op, x)
    op(x[1], x[2], x[3])
end

"""print the jacobian of one operator"""
function printone(op, syms, n)
    if n==1
        jac = jacobian_repeat(jf1, op, syms[1:1]; iin=2, iout=2)
    elseif n==2
        jac = jacobian_repeat(jf2, op, syms[1:2]; iin=2, iout=2)
    elseif n==3
        jac = jacobian_repeat(jf3, op, syms[1:3]; iin=2, iout=2)
    end
    println("------ $op ------")
    pretty_print_matrix(jac)
end

printall()

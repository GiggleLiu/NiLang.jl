using SymEngine
using SymEngine: BasicType

sconj = SymFunction("conj")
Base.conj(x::Basic) = Basic(conj(SymEngine.BasicType(x)))
Base.conj(x::BasicType) = real(x) - im * imag(x)
Base.imag(x::BasicType{Val{:Constant}}) = Basic(0)
Base.imag(x::BasicType{Val{:Symbol}}) = Basic(0)

pretty_print_number(x; lengthonly=false) = pretty_print_number(stdout, x; lengthonly=lengthonly)
function pretty_print_number(io::IO, x; lengthonly=false)
    sx = string(x)
    lengthonly || print(io, sx)
    return length(sx)
end

function pretty_print_number(io::IO, x::AbstractFloat; lengthonly=false)
    closest_int = round(Int, x)
    if isapprox(x, closest_int, atol=1e-12)
        si = string(closest_int)
        lengthonly || print(io, si)
        return length(si)
    else
        sx = string(x)
        lengthonly || print(io, sx)
        return length(sx)
    end
end

function pretty_print_number(io::IO, x::Complex; atol::Real = 1e-12, lengthonly=false)
    l = 0
    if !isapprox(real(x), 0, atol=atol)
        l += pretty_print_number(io, real(x), lengthonly=lengthonly)
    end
    if !isapprox(imag(x), 0, atol=atol)
        if !isapprox(real(x), 0, atol=atol)
            lengthonly || print(imag(x) > 0 ? "+" : "")
            l += 1
        end
        l += pretty_print_number(io, imag(x), lengthonly=lengthonly)
        lengthonly || print(io, "I")
        l += 1
    else
        if isapprox(real(x), 0, atol=atol)
            lengthonly || print(io, "0")
            l += 1
        end
    end
    return l
end

pretty_print_matrix(m) = pretty_print_matrix(stdout, m)
function pretty_print_matrix(io::IO, m)
    minlen = maximum(pretty_print_number.(m, lengthonly=true))+1
    for i in 1:size(m,1)
        print(io, "[")
        for j in 1:size(m,2)
            l = pretty_print_number(m[i,j])
            print(" "^(minlen-l-(j==size(m,1))))
        end
        println(io, "]")
    end
end

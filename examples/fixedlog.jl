using FixedPointNumbers, Test

"""
Reference
-------------------

[1] C. S. Turner,  "A Fast Binary Logarithm Algorithm", IEEE Signal
     Processing Mag., pp. 124,140, Sep. 2010.
"""
function log2fix(x::Fixed{T, P}) where {T, P}
    PREC = UInt(P)
    x.i == 0 && return typemin(T) # represents negative infinity

    y = zero(T)
    xi = x.i
    while xi < 1 << PREC
        xi <<= 1
        y -= T(1) << PREC
    end

    while xi >= 2 << PREC
        xi >>= 1
        y += T(1) << PREC
    end

    z = xi
    b = T(1) << (PREC - UInt(1))
    for i = 1:P
        temp = Base.widemul(z, z) >> PREC
        z = T(temp)
        if z >= T(2) << PREC
            z >>= 1
            y += b
        end
        b >>= 1
    end

    return Fixed{T,PREC}(y, nothing)
end

@test log2fix(Fixed{Int, 43}(2^1.24)) ≈ 1.24

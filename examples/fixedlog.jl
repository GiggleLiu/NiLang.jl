"""
## Reference
[1] C. S. Turner,  "A Fast Binary Logarithm Algorithm", IEEE Signal
     Processing Mag., pp. 124,140, Sep. 2010.
"""
function log2fix(x::Fixed{T, P}) where {T, P}
    PREC = UInt(P)
    x.i == 0 && return typemin(T) # represents negative infinity

    y = zero(T)
    xi = unsigned(x.i)
    while xi < UInt(1) << PREC
        xi <<= UInt(1)
        y -= 1 << PREC
    end

    while xi >= UInt(2) << PREC
        xi >>= UInt(1)
        y += 1 << PREC
    end

    z = Int128(xi)
    b = 1 << (PREC - UInt(1))
    for i = 1:P
        z = (z * z) >> PREC
        if z >= 2 << PREC
            z >>= UInt(1)
            y += b
        end
        b >>= UInt(1)
    end

    return Fixed{T,PREC}(y, nothing)
end

@test log2fix(Fixed43(2^1.24)) ≈ 1.24

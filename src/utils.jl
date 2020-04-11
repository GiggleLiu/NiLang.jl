export rot, plshift, prshift, arshift

"""
    rot(a, b, θ)

rotate variables `a` and `b` by an angle `θ`
"""
function rot(a, b, θ)
    s, c = sincos(θ)
    a*c-b*s, a*s+b*c
end

"""
    plshift(x, n)

periodic left shift.
"""
plshift(x, n) = (x << n) | (x >> (sizeof(x)*8-n))

"""
    plshift(x, n)

periodic right shift.
"""
prshift(x, n) = (x >> n) | (x << (sizeof(x)*8-n))

"""
    arshift(x, n)

right shift, sign extending.
"""
arshift(x::T, n) where T = (x >> n) | (x & (T(1) << (sizeof(x)*8-1)))

export unsafe_swap!, rot, plshift, prshift, arshift
function unsafe_swap!(tape, i, j)
    @inbounds temp = tape[i]
    @inbounds tape[i] = tape[j]
    @inbounds tape[j] = temp
end

function rot(a, b, θ)
    c = cos(θ)
    s = sin(θ)
    a*c-b*s, a*s+b*c
end
plshift(x, n) = (x << n) | (x >> (sizeof(x)*8-n))
prshift(x, n) = (x >> n) | (x << (sizeof(x)*8-n))
"""right shift, sign extending."""
arshift(x::T, n) where T = (x >> n) | (x & (T(1) << (sizeof(x)*8-1)))

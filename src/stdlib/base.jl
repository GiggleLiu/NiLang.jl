export i_sqdistance, i_dirtymul, i_factorial

"""
    i_sqdistance(dist!, x1, x2)

Squared distance between two points `x1` and `x2`.
"""
@i function i_sqdistance(dist!, x1::AbstractVector{T}, x2::AbstractVector) where T
    @inbounds for i=1:length(x1)
        x1[i] -= x2[i]
        dist! += x1[i] ^ 2
        x1[i] += x2[i]
    end
end

"""
    i_dirtymul(out!, x, anc!)

"dirty" reversible multiplication that computes `out! *= x` approximately for floating point numbers,
the `anc!` is anticipated as a number ~0.
"""
@i @inline function i_dirtymul(out!, x, anc!)
    anc! += out! * x
    out! -= anc! / x
    SWAP(out!, anc!)
end

@i @inline function i_dirtymul(out!::Int, x::Int, anc!::Int)
    anc! += out! * x
    out! -= anc! ÷ x
    SWAP(out!, anc!)
end

"""
    i_factorial(out!, n)

Compute the factorial `out! = factorial(n)`.
"""
@i function i_factorial(out!::Int, n::Int)
    INC(out!)
    @invcheckoff for i=1:n
        i_dirtymul(out!, i, 0)
    end
end

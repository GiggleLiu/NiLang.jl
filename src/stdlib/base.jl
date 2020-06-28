"""
    sqdistance(dist!, x1, x2)

Squared distance between two points `x1` and `x2`.
"""
@i function sqdistance(dist!, x1::AbstractVector{T}, x2::AbstractVector) where T
    @inbounds for i=1:length(x1)
        x1[i] -= identity(x2[i])
        dist! += x1[i] ^ 2
        x1[i] += identity(x2[i])
    end
end

"""
    imul(out!, x, anc!)

Reversible multiplication.
"""
@i @inline function imul(out!, x, anc!)
    anc! += out! * x
    out! -= anc! / x
    SWAP(out!, anc!)
end

@i @inline function imul(out!::Int, x::Int, anc!::Int)
    anc! += out! * x
    out! -= anc! ÷ x
    SWAP(out!, anc!)
end

@i function ifactorial(out!::Int, n::Int)
    INC(out!)
    @invcheckoff for i=1:n
        imul!(out!, i, 0)
    end
end




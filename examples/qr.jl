# # A simple QR decomposition

# ## Functions used in this example

using NiLang, NiLang.AD

"""
get the summation of an array.
"""
@i function isum(out!, x::AbstractArray)
    for i=1:length(x)
        out! += x[i]
    end
end


"""
dot product.
"""
@i function dot(out!, v1::Vector{T}, v2) where T
    for i = 1:length(v1)
        out! += v1[i]'*v2[i]
    end
end

"""
squared norm.
"""
@i function norm2(out!, vec::Vector{T}) where T
    anc1 ← zero(T)
    for i = 1:length(vec)
        anc1 += vec[i]'
        out! += anc1*vec[i]
        anc1 -= vec[i]'
    end
end

# ## The QR decomposition
# Let us consider a naive implementation of QR decomposition from scratch.
# This implementation is just a proof of principle which does not consider reorthogonalization and other practical issues.

@i function qr(Q, R, A::Matrix{T}) where T
    anc_norm ← zero(T)
    anc_dot ← zeros(T, size(A,2))
    ri ← zeros(T, size(A,1))
    for col = 1:size(A, 1)
        ri .+= A[:,col]
        for precol = 1:col-1
            dot(anc_dot[precol], Q[:,precol], ri)
            R[precol,col] += anc_dot[precol]
            for row = 1:size(Q,1)
                ri[row] -=
                    anc_dot[precol] * Q[row, precol]
            end
        end
        norm2(anc_norm, ri)

        R[col, col] += anc_norm^0.5
        for row = 1:size(Q,1)
            Q[row,col] += ri[row] / R[col, col]
        end

        ~begin
            ri .+= A[:,col]
            for precol = 1:col-1
                dot(anc_dot[precol], Q[:,precol], ri)
                for row = 1:size(Q,1)
                    ri[row] -= anc_dot[precol] *
                        Q[row, precol]
                end
            end
            norm2(anc_norm, ri)
        end
    end
end

# Here, in order to avoid frequent uncomputing, we allocate ancillas `ri` and `anc_dot` as vectors.
# The expression in `~` is used to uncompute `ri`, `anc_dot` and `anc_norm`.
# `dot` and `norm2` are reversible functions to compute dot product and vector norm.
# One can quickly check the correctness of the gradient function

A  = randn(4,4)
q, r = zero(A), zero(A)
@i function test1(out, q, r, A)
    qr(q, r, A)
    isum(out, q)
end

check_grad(test1, (0.0, q, r, A); iloss=1)

# Here, the loss function `test1` is defined as the sum of the output unitary matrix `q`.
# The `check_grad` function is a gradient checker function defined in module `NiLang.AD`.

# # A simple QR decomposition

# ## Functions used in this example

using NiLang, NiLang.AD, Test

# ## The QR decomposition
# Let us consider a naive implementation of QR decomposition from scratch.
# This implementation is just a proof of principle which does not consider reorthogonalization and other practical issues.

@i function qr(Q, R, A::Matrix{T}) where T
    @routine begin
        anc_norm ← zero(T)
        anc_dot ← zeros(T, size(A,2))
        ri ← zeros(T, size(A,1))
    end
    for col = 1:size(A, 1)
        ri .+= A[:,col]
        for precol = 1:col-1
            i_dot(anc_dot[precol], Q[:,precol], ri)
            R[precol,col] += anc_dot[precol]
            for row = 1:size(Q,1)
                ri[row] -=
                    anc_dot[precol] * Q[row, precol]
            end
        end
        i_norm2(anc_norm, ri)

        R[col, col] += anc_norm^0.5
        for row = 1:size(Q,1)
            Q[row,col] += ri[row] / R[col, col]
        end

        ~begin
            ri .+= A[:,col]
            for precol = 1:col-1
                i_dot(anc_dot[precol], Q[:,precol], ri)
                for row = 1:size(Q,1)
                    ri[row] -= anc_dot[precol] *
                        Q[row, precol]
                end
            end
            i_norm2(anc_norm, ri)
        end
    end
    ~@routine
end

# Here, in order to avoid frequent uncomputing, we allocate ancillas `ri` and `anc_dot` as vectors.
# The expression in `~` is used to uncompute `ri`, `anc_dot` and `anc_norm`.
# `i_dot` and `i_norm2` are reversible functions to compute dot product and vector norm.
# One can quickly check the correctness of the gradient function

A  = randn(4,4)
q, r = zero(A), zero(A)
@i function test1(out, q, r, A)
    qr(q, r, A)
    i_sum(out, q)
end

@test check_grad(test1, (0.0, q, r, A); iloss=1)

# Here, the loss function `test1` is defined as the sum of the output unitary matrix `q`.
# The `check_grad` function is a gradient checker function defined in module `NiLang.AD`.

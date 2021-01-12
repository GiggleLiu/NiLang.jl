# # Sparse matrices
#
# Source to source automatic differentiation is useful in differentiating sparse matrices. It is a well-known problem that sparse matrix operations can not benefit directly from generic backward rules for dense matrices because general rules do not keep the sparse structure.
# In the following, we will show that reversible AD can differentiate the Frobenius dot product between two sparse matrices with the state-of-the-art performance. Here, the Frobenius dot product is defined as \texttt{trace(A'B)}.
# Its native Julia (irreversible) implementation is `SparseArrays.dot`.
#
# The following is a reversible counterpart

using NiLang, NiLang.AD
using SparseArrays

@i function idot(r::T, A::SparseMatrixCSC{T},B::SparseMatrixCSC{T}) where {T}
    m ← size(A, 1)
    n ← size(A, 2)
    @invcheckoff branch_keeper ← zeros(Bool, 2*m)
    @safe size(B) == (m,n) || throw(DimensionMismatch("matrices must have the same dimensions"))
    @invcheckoff @inbounds for j = 1:n
        ia1 ← A.colptr[j]
        ib1 ← B.colptr[j]
        ia2 ← A.colptr[j+1]
        ib2 ← B.colptr[j+1]
        ia ← ia1
        ib ← ib1
        @inbounds for i=1:ia2-ia1+ib2-ib1-1
            ra ← A.rowval[ia]
            rb ← B.rowval[ib]
            if (ra == rb, ~)
                r += A.nzval[ia]' * B.nzval[ib]
            end
            ## b move -> true, a move -> false
            branch_keeper[i] ⊻= @const ia == ia2-1 || ra > rb
            ra → A.rowval[ia]
            rb → B.rowval[ib]
            if (branch_keeper[i], ~)
                INC(ib)
            else
                INC(ia)
            end
        end
        ~@inbounds for i=1:ia2-ia1+ib2-ib1-1
            ## b move -> true, a move -> false
            branch_keeper[i] ⊻= @const ia == ia2-1 || A.rowval[ia] > B.rowval[ib]
            if (branch_keeper[i], ~)
                INC(ib)
            else
                INC(ia)
            end
        end
    end
    @invcheckoff branch_keeper → zeros(Bool, 2*m)
end

# Here, the key point is using a \texttt{branch\_keeper} vector to cache branch decisions.

# The time used for a native implementation is

using BenchmarkTools
a = sprand(1000, 1000, 0.01);
b = sprand(1000, 1000, 0.01);
@benchmark SparseArrays.dot($a, $b)

# To compute the gradients, we wrap each matrix element with `GVar`, and send them to the reversible backward pass

out! = SparseArrays.dot(a, b)
@benchmark (~idot)($(GVar(out!, 1.0)),
        $(GVar.(a)), $(GVar.(b)))

# The time used for computing backward pass is approximately 1.6 times Julia's native forward pass.
# Here, we have turned off the reversibility check off to achieve better performance.
# By writing sparse matrix multiplication and other sparse matrix operations reversibly,
# we will have a differentiable sparse matrix library with proper performance.

# See my another blog post for [reversible sparse matrix multiplication](https://nextjournal.com/giggle/how-to-write-a-program-differentiably).

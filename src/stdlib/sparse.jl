using SparseArrays

@i function i_mul!(C::StridedVecOrMat, A::AbstractSparseMatrix, B::StridedVector{T}, α::Number, β::Number) where T
    @safe size(A, 2) == size(B, 1) || throw(DimensionMismatch())
    @safe size(A, 1) == size(C, 1) || throw(DimensionMismatch())
    @safe size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    nzv ← nonzeros(A)
    rv ← rowvals(A)
    if (β != 1, ~)
        @safe error("only β = 1 is supported, got β = $(β).")
    end
    # Here, we close the reversibility check inside the loop to increase performance
    @invcheckoff for k = 1:size(C, 2)
        @inbounds for col = 1:size(A, 2)
            αxj ← zero(T)
            αxj += B[col,k] * α
            for j = SparseArrays.getcolptr(A)[col]:(SparseArrays.getcolptr(A)[col + 1] - 1)
                C[rv[j], k] += nzv[j]*αxj
            end
            αxj -= B[col,k] * α
        end
    end
end


@i function i_dot(r::T, A::SparseMatrixCSC{T},B::SparseMatrixCSC{T}) where {T}
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
            branch_keeper[i] ⊻= @const ia == ia2-1 || (ib != ib2-1 && ra > rb)
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
            branch_keeper[i] ⊻= @const ia == ia2-1 || (ib != ib2-1 && A.rowval[ia] > B.rowval[ib])
            if (branch_keeper[i], ~)
                INC(ib)
            else
                INC(ia)
            end
        end
    end
    @invcheckoff branch_keeper → zeros(Bool, 2*m)
end

using NiLang, NiLang.AD
using KernelAbstractions, CUDA

@i @kernel function kernel_f(A, B::AbstractVector{TB}) where TB
    # turng off reversibility check, since GPU can not handle errors
    @invcheckoff begin
        # allocate
        batch ← @index(Global)
        s ← zero(TB)
        # computing
        for i in axes(A, 1)
            s += A[i, i, batch]
        end
        B[batch] += s
        # deallocate safely
        s → zero(TB)
        batch → @index(Global)
    end
end

@i function batched_tr!(A::CuArray{T, 3}, B::CuVector{T}) where T
    @launchkernel CUDADevice() 256 length(B) kernel_f(A, B)
end

A = CuArray(randn(ComplexF32, 10, 10, 100))
B = CUDA.zeros(ComplexF32, 100)
A_out, B_out = batched_tr!(A, B)
# put random values in the gradient field of B
grad_B = CuArray(randn(ComplexF32, 100))
A_with_g, B_with_g = (~batched_tr!)(GVar(A_out), GVar(B_out, grad_B))
# will see nonzero gradients in complex diagonal parts of A
grad_A = grad(A_with_g |> Array)

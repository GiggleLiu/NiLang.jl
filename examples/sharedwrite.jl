# # The shared write problem on GPU

# We will write a GPU version of `axpy!` function.

# ## The main program

using KernelAbstractions
using NiLang, NiLang.AD
using CuArrays

# so far, this example requires patch: https://github.com/JuliaGPU/KernelAbstractions.jl/pull/52

@i @kernel function axpy_kernel(y!, α, x)
    ## invcheckoff to turn of `reversibility checker`
    ## GPU can not handle errors!
    @invcheckoff begin
        i ← @index(Global)
        y![i] += x[i] * α
        i → @index(Global)
    end
end

@i function cu_axpy!(y!::AbstractVector, α, x::AbstractVector)
    @launchkernel CUDA() 256 length(y!) axpy_kernel(y!, α, x)
end

@i function loss(out, y!, α, x)
    cu_axpy!(y!, α, x)
    ## Note: the following code is stupid scalar operations on CuArray,
    ## They are only for testing.
    for i=1:length(y!)
        out += identity(y![i])
    end
end

y! = rand(100)
x = rand(100)
cuy! = y! |> CuArray
cux = x |> CuArray
α = 0.4

# ## Check the correctness of results

using Test
cu_axpy!(cuy!, α, cux)
@test cuy! ≈ y! .+ α .* x
(~cu_axpy!)(cuy!, α, cux)
@test cuy! ≈ y!

# Let's check the gradients
lsout = 0.0
@instr loss'(Val(1), lsout, cuy!, α, cux)

# you will see a correct vector `[0.4, 0.4, 0.4 ...]`
grad.(cux)

# you will see `0.0`.
grad(α)

# ## Why some gradients not correct?
# In the above example, `α` is a scalar, whereas a scalar is not allowed to change in a CUDA kernel.
# What if we change `α` to a CuArray?

# ## This one works: using a vector of `α`
@i @kernel function axpy_kernel(y!, α, x)
    @invcheckoff begin
        i ← @index(Global)
        y![i] += x[i] * α[i]
        i → @index(Global)
    end
end

cuy! = y! |> CuArray
cux = x |> CuArray
cuβ = repeat([0.4], 100) |> CuArray
lsout = 0.0
@instr loss'(Val(1), lsout, cuy!, cuβ, cux)

# You will see correct answer
grad.(cuβ)

# ## This one has the shared write problem: using a vector of `α`, but shared read.
@i @kernel function axpy_kernel(y!, α, x)
    @invcheckoff begin
        i ← @index(Global)
        y![i] += x[i] * α[i]
        i → @index(Global)
    end
end

cuy! = y! |> CuArray
cux = x |> CuArray
cuβ = repeat([0.4], 100) |> CuArray
lsout = 0.0
cuβ = [0.4] |> CuArray

# Run the following will give you a happy error
#
# > ERROR: a exception was thrown during kernel execution.
# >        Run Julia on debug level 2 for device stack traces.

# ```julia
# @instr loss'(Val(1), lsout, cuy!, cuβ, cux)
# ```

# Because, shared write is not allowed. We need someone clever enough to solve this problem for us.

# ## Conclusion
# * Shared scalar: the gradient of a scalar will not be updated.
# * Expanded vector: works properly, but costs more memory.
# * Shared 1-element vector: error on shared write.

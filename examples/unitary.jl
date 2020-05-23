# # Unitary matrix operations without allocation
# A unitary matrix features uniform eigenvalues and reversibility. It is widely used as an approach to ease the gradient exploding and vanishing problem and the memory wall problem.
# One of the simplest ways to parametrize a unitary matrix is representing a unitary matrix as a product of two-level unitary operations. A real unitary matrix of size $N$ can be parametrized compactly by $N(N-1)/2$ rotation operations
#
# ```math
#    {\rm ROT}(a!, b!, \theta)  = \left(\begin{matrix}
#        \cos(\theta) & - \sin(\theta)\\
#        \sin(\theta)  & \cos(\theta)
#    \end{matrix}\right)
#    \left(\begin{matrix}
#        a!\\
#        b!
#    \end{matrix}\right),
# ```
#
# where $\theta$ is the rotation angle, `a!` and `b!` are target registers.

using NiLang, NiLang.AD

@i function umm!(x!, θ)
    @safe @assert length(θ) ==
            length(x!)*(length(x!)-1)/2
    k ← 0
    for j=1:length(x!)
        for i=length(x!)-1:-1:j
            k += identity(1)
            ROT(x![i], x![i+1], θ[k])
        end
    end

    k → length(θ)
end

# Here, the ancilla `k` is deallocated manually by specifying its value, because we know the loop size is $N(N-1)/2$.
# We define the test functions in order to check gradients.

@i function isum(out!, x::AbstractArray)
    for i=1:length(x)
        out! += identity(x[i])
    end
end

@i function test!(out!, x!::Vector, θ::Vector)
   umm!(x!, θ)
   isum(out!, x!)
end

# Let's print the program output

out, x, θ = 0.0, randn(4), randn(6);
@instr Grad(test!)(Val(1), out, x, θ)
x

# We can erease the gradient field by uncomputing the gradient function.
# If you want, you can differentiate it twice to obtain Hessians.
# However, we suggest using ForwardDifferentiation over our NiLang program, this is more efficient.

@instr (~Grad(test!))(Val(1), out, x, θ)
x

# In the above testing code, `Grad(test)` attaches a gradient field to each element of `x`. `~Grad(test)` is the inverse program that erase the gradient fields.
# Notably, this reversible implementation costs zero memory allocation, although it changes the target variables inplace.

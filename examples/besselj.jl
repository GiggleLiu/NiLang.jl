# # Bessel function
# An Bessel function of the first kind of order ``\nu`` can be computed using Taylor expansion

# ```math
#     J_\nu(z) = \sum\limits_{n=0}^{\infty} \frac{(z/2)^\nu}{\Gamma(k+1)\Gamma(k+\nu+1)} (-z^2/4)^{n}
# ```

# where ``\Gamma(n) = (n-1)!`` is the Gamma function. One can compute the accumulated item iteratively as ``s_n = -\frac{z^2}{4} s_{n-1}``.
# Intuitively, this problem mimics the famous pebble game, since one can not release state ``s_{n-1}`` directly after computing ``s_n``.
# One would need an increasing size of tape to cache the intermediate state.
# To circumvent this problem. We introduce the following reversible approximate multiplier

using NiLang, NiLang.AD

# Here, the definition of SWAP can be found in \App{app:instr}, ``anc! \approx 0`` is a *dirty ancilla*.
# Line 2 computes the result and accumulates it to the dirty ancilla, we get an approximately correct output in **anc!**.
# Line 3 "uncomputes" **out!** approximately by using the information stored in **anc!**, leaving a dirty zero state in register **out!**.
# Line 4 swaps the contents in **out!** and **anc!**.
# Finally, we have an approximately correct output and a dirtier ancilla.
# With this multiplier, we implementation ``J_\nu`` as follows.

@i function ibesselj(out!::T, ν, z::T; atol=1e-8) where T
    @routine @invcheckoff begin
        k ← 0
        fact_nu ← zero(ν)
        @zeros T halfz halfz_power_nu halfz_power_2 out_anc anc1 anc2 anc3 anc4 anc5
        halfz += z / 2
        halfz_power_nu += halfz ^ ν
        halfz_power_2 += halfz ^ 2
        i_factorial(fact_nu, ν)
        anc1 += halfz_power_nu/fact_nu
        out_anc += anc1
        while (abs(unwrap(anc1)) > atol && abs(unwrap(anc4)) < atol, k!=0)
            INC(k)
            @routine begin
                anc5 += k + ν
                anc2 -= k * anc5
                anc3 += halfz_power_2 / anc2
            end
            i_dirtymul(anc1, anc3, anc4)
            out_anc += anc1
            ~@routine
        end
    end
    out! += out_anc
    ~@routine
end

# where the **i_factorial** is defined as

# Here, only a constant number of ancillas are used in this implementation, while the algorithm complexity does not increase comparing to its irreversible counterpart.
# ancilla **anc4** plays the role of *dirty ancilla* in multiplication, it is uncomputed rigoriously in the uncomputing stage.
# The reason why the "approximate uncomputing" trick works here lies in the fact that from the mathematic perspective the state in ``n``th step ``\{s_n, z\}`` contains the same amount of information as the state in the ``n-1``th step ``\{s_{n-1}, z\}`` except some special points, it is highly possible to find an equation to uncompute the previous state from the current state.
# This trick can be used extensively in many other application. It mitigated the artifitial irreversibility brought by the number system that we have adopt at the cost of precision.

# To obtain gradients, one call **Grad(ibesselj)**

y, x = 0.0, 1.0
Grad(ibesselj)(Val(1), y, 2, x)

# Here, **Grad(ibesselj)** is a callable instance of type **Grad{typeof(ibesselj)}}**.
# The first parameter `Val(1)` indicates the first argument is the loss.

# To obtain second order gradients, one can Feed dual numbers to this gradient function.
using ForwardDiff: Dual
_, hxy, _, hxx = Grad(ibesselj)(Val(1), Dual(y, zero(y)), 2, Dual(x, one(x)))
println("The hessian dy^2/dx^2 is $(grad(hxx).partials[1])")

# Here, the gradient field is a Dual number, it has a field partials that stores the derivative with respect to `x`.
# This is the Hessian that we need.

# ## CUDA programming
# The AD in NiLang avoids most heap allocation, so that it is able to execute on a GPU device
# We suggest using [KernelAbstraction](https://github.com/JuliaGPU/KernelAbstractions.jl), it provides compatibility between CPU and GPU.
# To execute the above function on GPU, we need only 11 lines of code.

# ```julia
# using CuArrays, GPUArrays, KernelAbstractions
#
# @i @kernel function bessel_kernel(out!, v, z)
#     @invcheckoff i ← @index(Global)
#     ibesselj(out![i], v, z[i])
#     @invcheckoff i → @index(Global)
# end
# ```

# We have a macro support to KernelAbstraction in NiLang.
# So it is possible to launch directly like.
# ```julia
# @i function befunc(out!, v::Integer, z)
#     @launchkernel CUDA() 256 length(out!) bessel_kernel(out!, v, z)
# end
# ```

# It is equivalent to call
# ```julia
# (~bessel_kernel)(CUDA(), 256)(out!, v, z; ndrange=length(out!))
# ```
# But it will execute the job eagerly for you.
# We will consider better support in the future.

# Except it is reversible
# ```julia repl
# julia> @code_reverse @launchkernel CUDA() 256 length(out!) bessel_kernel(out!, v, z)
# :(#= REPL[4]:1 =# @launchkernel CUDA() 256 length(out!) (~bessel_kernel)(out!, v, z))
# ```

# To test this function, we first define input parameters `a` and output `out!`
# ```julia
# a = CuArray(rand(128))
# out! = CuArray(zeros(128))
# ```

# We wrap the output with a randomly initialized gradient field, suppose we get the gradients from a virtual loss function.
# Also, we need to initialize an empty gradient field for elements in input cuda tensor `a`.
# ```julia
# out! = ibesselj(out!, 2, GVar.(a))[1]
# out_g! = GVar.(out!, CuArray(randn(128)))
# ```

# Call the inverse program, the multiple dispatch will drive you to the goal.
# ```julia
# (~ibesselj)(out_g!, 2, GVar.(a))
# ```

# You will get CUDA arrays with `GVar` elements as output, their gradient fields are what you want.
# Cheers! Now you have a adjoint mode differentiable CUDA kernel.

# ## Benchmark
# We have different source to souce automatic differention implementations of the first type Bessel function ``J_2(1.0)`` benchmarked and show the results below.
#
#
# |  Package  | Tangent/Adjoint | ``T_{\rm min}``/ns  |  Space/KB |
# | --------- | --------------- | ------------------- | --------- |
# |  Julia    |     -           |     22              |     0     |
# |  NiLang   |     -           |     59              |     0     |
# |  ForwardDiff |    Tangent   |     35              |     0     |
# |  Manual   |    Adjoint      |     83              |     0     |
# |  NiLang.AD |    Adjoint     |     213             |     0     |
# |  NiLang.AD (GPU) | Adjoint  |     1.4             |     0     |
# |  Zygote   |    Adjoint      |      31201          |   13.47   |
# |  Tapenade |    Adjoint      |     ?               |     ?     |

# Julia is the CPU time used for running the irreversible forward program, is the baseline of benchmarking.
# NiLang is the reversible implementation, it is 2.7 times slower than its irreversible counterpart. Here, we have remove the reversibility check.
# ForwardDiff gives the best performance because it is designed for functions with single input.
# It is even faster than manually derived gradients
# ```math
# \frac{\partial J_{\nu}(z)}{\partial z} = \frac{J_{\nu-1} - J_{\nu+1}}{2}
# ```
# NiLang.AD is the reversible differential programming implementation, it considers only the backward pass.
# The benchmark of its GPU version is estimated on Nvidia Titan V by broadcasting the gradient function on CUDA array of size ``2^17`` and take average.
# The Zygote benchmark considers both forward pass and backward pass.
# Tapenade is not yet ready.

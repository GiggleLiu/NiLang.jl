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

@i @inline function imul(out!, x, anc!)
    anc! += out! * x
    out! -= anc! / x
    SWAP(out!, anc!)
end

# Here, the definition of SWAP can be found in \App{app:instr}, ``anc! \approx 0`` is a *dirty ancilla*.
# Line 2 computes the result and accumulates it to the dirty ancilla, we get an approximately correct output in **anc!**.
# Line 3 "uncomputes" **out!** approximately by using the information stored in **anc!**, leaving a dirty zero state in register **out!**.
# Line 4 swaps the contents in **out!** and **anc!**.
# Finally, we have an approximately correct output and a dirtier ancilla.
# With this multiplier, we implementation ``J_\nu`` as follows.

@i function ibesselj(out!, ν, z; atol=1e-8)
    @routine @invcheckoff begin
        k ← 0
        fact_nu ← zero(ν)
        halfz ← zero(z)
        halfz_power_nu ← zero(z)
        halfz_power_2 ← zero(z)
        out_anc ← zero(z)
        anc1 ← zero(z)
        anc2 ← zero(z)
        anc3 ← zero(z)
        anc4 ← zero(z)
        anc5 ← zero(z)

        halfz += z / 2
        halfz_power_nu += halfz ^ ν
        halfz_power_2 += halfz ^ 2
        ifactorial(fact_nu, ν)

        anc1 += halfz_power_nu/fact_nu
        out_anc += identity(anc1)
        while (abs(unwrap(anc1)) > atol && abs(unwrap(anc4)) < atol, k!=0)
            k += identity(1)
            @routine begin
                anc5 += identity(k)
                anc5 += identity(ν)
                anc2 -= k * anc5
                anc3 += halfz_power_2 / anc2
            end
            imul(anc1, anc3, anc4)
            out_anc += identity(anc1)
            ~@routine
        end
    end
    out! += identity(out_anc)
    ~@routine
end

# where the **ifactorial** is defined as

@i function ifactorial(out!, n)
    out! += identity(1)
    for i=1:n
        MULINT(out!, i)
    end
end


# Here, only a constant number of ancillas are used in this implementation, while the algorithm complexity does not increase comparing to its irreversible counterpart.
# ancilla **anc4** plays the role of *dirty ancilla* in multiplication, it is uncomputed rigoriously in the uncomputing stage.
# The reason why the "approximate uncomputing" trick works here lies in the fact that from the mathematic perspective the state in ``n``th step ``\{s_n, z\}`` contains the same amount of information as the state in the ``n-1``th step ``\{s_{n-1}, z\}`` except some special points, it is highly possible to find an equation to uncompute the previous state from the current state.
# This trick can be used extensively in many other application. It mitigated the artifitial irreversibility brought by the number system that we have adopt at the cost of precision.

# To obtain gradients, one can wrap the variable **y!** with **Loss** type and feed it into **ibesselj'**

y, x = 0.0, 3.0
ibesselj'(Loss(y), 2, x)

# Here, **ibesselj'** is a callable instance of type **Grad{typeof(ibesselj)}}**. This function itself is reversible and differentiable, one can back-propagate this function to obtain Hessians. In NiLang, it is implemented as **hessian_repeat**.

hessian_repeat(ibesselj, (Loss(y), 2, x))

# ## CUDA programming
# You need a patch to define the gradients for "CUDAnative.pow".

using NiLang, NiLang.AD
using CuArrays, CUDAnative, GPUArrays

@i @inline function ⊖(CUDAnative.pow)(out!::GVar{T}, x::GVar, n::GVar) where T
    ⊖(CUDAnative.pow)(value(out!), value(x), value(n))

    # grad x
    @routine @invcheckoff begin
        anc1 ← zero(value(x))
        anc2 ← zero(value(x))
        anc3 ← zero(value(x))
        jac1 ← zero(value(x))
        jac2 ← zero(value(x))

        value(n) -= identity(1)
        anc1 += CUDAnative.pow(value(x), value(n))
        value(n) += identity(1)
        jac1 += anc1 * value(n)

        # get grad of n
        anc2 += log(value(x))
        anc3 += CUDAnative.pow(value(x), value(n))
        jac2 += anc3*anc2
    end
    grad(x) += grad(out!) * jac1
    grad(n) += grad(out!) * jac2
    ~@routine
end

@i @inline function ⊖(CUDAnative.pow)(out!::GVar{T}, x::GVar, n) where T
    ⊖(CUDAnative.pow)(value(out!), value(x), n)
    @routine @invcheckoff begin
        anc1 ← zero(value(x))
        jac ← zero(value(x))

        value(n) -= identity(1)
        anc1 += CUDAnative.pow(value(x), n)
        value(n) += identity(1)
        jac += anc1 * n
    end
    grad(x) += grad(out!) * jac
    ~@routine
end

@i @inline function ⊖(CUDAnative.pow)(out!::GVar{T}, x, n::GVar) where T
    ⊖(CUDAnative.pow)(value(out!), x, value(n))
    # get jac of n
    @routine @invcheckoff begin
        anc1 ← zero(x)
        anc2 ← zero(x)
        jac ← zero(x)

        anc1 += log(x)
        anc2 += CUDAnative.pow(x, value(n))
        jac += anc1*anc2
    end
    grad(n) += grad(out!) * jac
    ~@routine
end


# You need to replace all "^" operations in `ibessel` with `CUDAnative.pow`.
# Please remember to turn invertiblity check off, because error handling is not supported in a cuda thread.
# Function `imul` and `ifactorial` are not changed.

@i function ibesselj(out!, ν, z; atol=1e-8)
    @routine @invcheckoff begin
        k ← 0
        fact_nu ← zero(ν)
        halfz ← zero(z)
        halfz_power_nu ← zero(z)
        halfz_power_2 ← zero(z)
        out_anc ← zero(z)
        anc1 ← zero(z)
        anc2 ← zero(z)
        anc3 ← zero(z)
        anc4 ← zero(z)
        anc5 ← zero(z)

        halfz += z / 2
        halfz_power_nu += CUDAnative.pow(halfz, ν)
        halfz_power_2 += CUDAnative.pow(halfz, 2)
        ifactorial(fact_nu, ν)

        anc1 += halfz_power_nu/fact_nu
        out_anc += identity(anc1)
        while (abs(unwrap(anc1)) > atol && abs(unwrap(anc4)) < atol, k!=0)
            k += identity(1)
            @routine begin
                anc5 += identity(k)
                anc5 += identity(ν)
                anc2 -= k * anc5
                anc3 += halfz_power_2 / anc2
            end
            imul(anc1, anc3, anc4)
            out_anc += identity(anc1)
            ~@routine
        end
    end
    out! += identity(out_anc)
    ~@routine
end

# Define your reversible kernel function that calls the reversible bessel function

@i function ibesselj_kernel(out!, ν, z, atol)
    i ← (blockIdx().x-1) * blockDim().x + threadIdx().x
    @inbounds ibesselj(out![i], ν, z[i]; atol=atol)
    @invcheckoff i → (blockIdx().x-1) * blockDim().x + threadIdx().x
end

# To launch this reversible kernel, you also need a reversible host function.

@i function ibesselj(out!::CuVector, ν, z::CuVector; atol=1e-8)
   XY ← GPUArrays.thread_blocks_heuristic(length(out!))
   @cuda threads=tget(XY,1) blocks=tget(XY,2) ibesselj_kernel(out!, ν, z, atol)
   @invcheckoff XY → GPUArrays.thread_blocks_heuristic(length(out!))
end

# To test this function, we first define input parameters `a` and output `out!`
a = CuArray(rand(128))
out! = CuArray(zeros(128))

# We wrap the output with a randomly initialized gradient field, suppose we get the gradients from a virtual loss function.
# Also, we need to initialize an empty gradient field for elements in input cuda tensor `a`.
out! = ibesselj(out!, 2, GVar.(a))[1]
out_g! = GVar.(out!, CuArray(randn(128)))

# Call the inverse program, the multiple dispatch will drive you to the goal.
(~ibesselj)(out_g!, 2, GVar.(a))

# You will get CUDA arrays with `GVar` elements as output, their gradient fields are what you want.
# Cheers! Now you have a adjoint mode differentiable CUDA kernel.

# ## Benchmark
# We have different source to souce automatic differention implementations of the first type Bessel function ``J_2(1.0)`` benchmarked and show the results below.
#
# 
# |           | Tangent/Adjoint | ``T_{\rm min}``/ns  |  Space/KB |
# | --------- | --------------- | ------------------- | --------- |
# |  Julia | - | 22 | 0 |
# |  NiLang | - | 59 | 0 |
# |  ForwardDiff | Tangent | 35 | 0 |
# |  Manual | Adjoint | 83 | 0 |
# |  NiLang.AD | Adjoint | 213 | 0 |
# |  NiLang.AD (GPU) | Adjoint | 1.4 | 0 |
# |  Zygote | Adjoint | 31201 | 13.47 |
# |  Tapenade | Adjoint | ? | ? |

# Julia is the CPU time used for running the irreversible forward program, is the baseline of benchmarking.
# NiLang is the reversible implementation, it is 2.7 times slower than its irreversible counterpart. Here, we have remove the reversibility check.
# ForwardDiff gives the best performance because it is designed for functions with single input.
# It is even faster than manually derived gradients
# ```math
# J_{\nu}'(z) = \frac{J_{\nu-1} - J_{\nu+1}}{2}
# ```
# NiLang.AD is the reversible differential programming implementation, it considers only the backward pass.
# The benchmark of its GPU version is estimated on Nvidia Titan V by broadcasting the gradient function on CUDA array of size ``2^17`` and take average.
# The Zygote benchmark considers both forward pass and backward pass.
# Tapenade is not yet ready.

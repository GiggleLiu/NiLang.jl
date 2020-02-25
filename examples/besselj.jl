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

    @routine begin
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

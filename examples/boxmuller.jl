# # Box-Muller method to Generate normal distribution
using NiLang

# In this tutorial, we introduce using Box-Muller method to transform a uniform distribution to a normal distribution.
# The transformation and inverse transformation of `Box-Muller` method could be found in
# [this blog](https://mathworld.wolfram.com/Box-MullerTransformation.html)
@i function boxmuller(x::T, y::T) where T
    @routine @invcheckoff begin
        θ ← zero(T)
        logx ← zero(T)
        _2logx ← zero(T)

        θ += 2π * y
        logx += log(x)
        _2logx += -2 * logx
    end

    ## store results
    z1 ← zero(T)
    z2 ← zero(T)
    z1 += _2logx ^ 0.5
    ROT(z1, z2, θ)
    ~@routine

    SWAP(x, z1)
    SWAP(y, z2)

    ## arithmetic uncomputing: recomputing the original values of `x` and `y` to deallocate z1 and z2
    @routine @invcheckoff begin
        at ← zero(T)
        sq ← zero(T)
        _halfsq ← zero(T)
        at += atan(y, x)
        if (y < 0, ~)
            at += T(2π)
        end
        sq += x ^ 2
        sq += y ^ 2
        _halfsq -= sq / 2
    end
    z1 -= exp(_halfsq)
    z2 -= at / (2π)
    @invcheckoff z1 → zero(T)
    @invcheckoff z2 → zero(T)
    ~@routine
end

# One may wonder why this implementation is so long,
# should't NiLang generate the inverse for user?
# The fact is, although Box-Muller is arithmetically reversible.
# It is not finite precision reversible.
# Hence we need to "uncompute" it manually,
# this trick may introduce reversibility error.

using Plots
N = 5000
x = rand(2*N)

Plots.histogram(x, bins = -3:0.1:3, label="uniform",
    legendfontsize=16, xtickfontsize=16, ytickfontsize=16)

# forward
@instr boxmuller.(x[1:N], x[N+1:end])
Plots.histogram(x, bins = -3:0.1:3, label="normal",
    legendfontsize=16, xtickfontsize=16, ytickfontsize=16)

# backward
@instr (~boxmuller).(x[1:N], x[N+1:end])
Plots.histogram(x, bins = -3:0.1:3, label="uniform",
    legendfontsize=16, xtickfontsize=16, ytickfontsize=16)

# ## Check the probability distribution function
using LinearAlgebra, Test

normalpdf(x) = sqrt(1/2π)*exp(-x^2/2)

# obtain `log(abs(det(jacobians)))`
@i function f(x::Vector)
    boxmuller(x[1], x[2])
end
jac = NiLang.AD.jacobian(f, [0.5, 0.5], iin=1)
ladj = log(abs(det(jac)))

# check if it matches the `log(p/q)`.
z1, z2 = boxmuller(0.5, 0.5)
@test ladj ≈ log(1.0 / (normalpdf(z1) * normalpdf(z2)))

# ## To obtaining Jacobian - a simpler approach
# We can define a function that exactly reversible from the instruction level,
# but costs more space for storing output.
@i function boxmuller2(x1::T, x2::T, z1::T, z2::T) where T
    @routine @invcheckoff begin
        θ ← zero(T)
        logx ← zero(T)
        _2logx ← zero(T)

        θ += 2π * x2
        logx += log(x1)
        _2logx += -2 * logx
    end

    ## store results
    z1 += _2logx ^ 0.5
    ROT(z1, z2, θ)
    ~@routine
end

# However, this is not a bijector from that maps `x` to `z`,
# because computing the backward just erases the content in `z`.
# However, this function can be used to obtain `log(abs(det(jacobians)))`
@i function f2(x::Vector, z::Vector)
    boxmuller2(x[1], x[2], z[1], z[2])
end
jac = NiLang.AD.jacobian(f2, [0.5, 0.5], [0.0, 0.0], iin=1, iout=2)
ladj = log(abs(det(jac)))

# check if it matches the `log(p/q)`.
_, _, z1, z2 = boxmuller2(0.5, 0.5, 0.0, 0.0)
@test ladj ≈ log(1.0 / (normalpdf(z1) * normalpdf(z2)))

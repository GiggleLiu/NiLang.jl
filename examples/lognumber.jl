# # Logarithmic number system

# Computing basic functions like `power`, `exp` and `besselj` is not trivial for reversible programming.
# There is no efficient constant memory algorithm using pure fixed point numbers only.
# For example, to compute `x ^ n` reversiblly with fixed point numbers,
# we need to allocate a vector of size $O(n)$.
# With logarithmic numbers, the above computation is straight forward.

using LogarithmicNumbers
using NiLang, NiLang.AD
using FixedPointNumbers

@i function i_power(y::T, x::T, n::Int) where T
    if !iszero(x)
        @routine begin
            lx ← one(ULogarithmic{T})
            ly ← one(ULogarithmic{T})
            ## convert `x` to a logarithmic number
            ## Here, `*=` is reversible for log numbers
            if x > 0
                lx *= convert(x)
            else
                lx *= convert(-x)
            end
            for i=1:n
                ly *= lx
            end
        end

        ## convert back to fixed point numbers
        y += convert(ly)
        if x < 0 && n%2 == 1
            NEG(y)
        end

        ~@routine
    end
end

# To check the function
i_power(Fixed43(0.0), Fixed43(0.4), 3)

# ## `exp` function as an example
# The following example computes `exp(x)`.

@i function i_exp(y!::T, x::T) where T<:Union{Fixed, GVar{<:Fixed}}
    @invcheckoff begin
        @routine begin
            s ← one(ULogarithmic{T})
            lx ← one(ULogarithmic{T})
            k ← 0
        end
        lx *= convert(x)
        y! += convert(s)
        @from k==0 while s.log > -20
            k += 1
            s *= lx / k
            y! += convert(s)
        end
        ~(@from k==0 while s.log > -20
            k += 1
            s *= x / k
        end)
        lx /= convert(x)
        ~@routine
    end
end

x = Fixed43(3.5)

# We can check the reversibility
out, _ = i_exp(Fixed43(0.0), x)
@assert out ≈ exp(3.5)

# Computing the gradients
_, gx = NiLang.AD.gradient(Val(1), i_exp, (Fixed43(0.0), x))
@assert gx ≈ exp(3.5)

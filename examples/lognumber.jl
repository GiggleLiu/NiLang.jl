using LogarithmicNumbers
using NiLang, NiLang.AD
using FixedPointNumbers

@i function i_exp(y!::T, x::T) where T<:Union{Fixed, GVar{<:Fixed}}
    @invcheckoff begin
        @routine begin
            s ← one(ULogarithmic{T})
            lx ← one(ULogarithmic{T})
            k ← 0
        end
        lx *= convert(x)
        y! += convert(s)
        while (s.log > -20, k != 0)
            k += 1
            s *= lx / k
            y! += convert(s)
        end
        ~(while (s.log > -20, k != 0)
            k += 1
            s *= x / k
        end)
        lx /= convert(x)
        ~@routine
    end
end

x = Fixed43(3.5)
i_exp(Fixed43(0.0), x)
Grad(i_exp)(Val(1), Fixed43(0.0), x)

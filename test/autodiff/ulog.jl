using NiLang, NiLang.AD
using Test, Random
using ForwardDiff
using FixedPointNumbers
using NiLangCore: default_constructor

@testset "ULogarithmic" begin
	@test check_grad(PlusEq(gaussian_log), (1.0, 2.0); iloss=1)
	function muleq(f, x::T, y::T, z::T) where T
        x = default_constructor(ULogarithmic{T}, x)
        y = default_constructor(ULogarithmic{T}, y)
        z = default_constructor(ULogarithmic{T}, z)
		x *= f(y, z)
		x.log
	end
	g1 = ForwardDiff.gradient(arr->muleq(+, arr...), [7.0, 5.0, 3.0])
    x, y, z = default_constructor(ULogarithmic{Float64}, 7.0),
    default_constructor(ULogarithmic{Float64}, 5.0),
    default_constructor(ULogarithmic{Float64}, 3.0)
	@instr (MulEq(+))(x, y, z)
	@instr GVar(x)
	@instr GVar(y)
	@instr GVar(z)
	@instr x.log.g += 1
	@instr (~MulEq(+))(x, y, z)
	@test grad(x.log) ≈ g1[1]
	@test grad(y.log) ≈ g1[2]
	@test grad(z.log) ≈ g1[3]

	g2 = ForwardDiff.gradient(arr->muleq(-, arr...), [7.0, 5.0, 3.0])
    x, y, z = default_constructor(ULogarithmic{Float64}, 2.0),
    default_constructor(ULogarithmic{Float64}, 5.0),
    default_constructor(ULogarithmic{Float64}, 3.0)
	@instr (MulEq(-))(x, y, z)
	@instr GVar(x)
	@instr GVar(y)
	@instr GVar(z)
	@instr x.log.g += 1
	@instr (~MulEq(-))(x, y, z)
	@test grad(x.log) ≈ g2[1]
	@test grad(y.log) ≈ g2[2]
	@test grad(z.log) ≈ g2[3]
end

@testset "iexp" begin
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
	res = i_exp(Fixed43(0.0), x)[1]
	gx = grad(Grad(i_exp)(Val(1), Fixed43(0.0), x)[3])
	@test res ≈ exp(3.5)
	@test gx ≈ exp(3.5)
end

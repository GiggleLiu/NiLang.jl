using NiLang, NiLang.AD
using Test, Random
using ForwardDiff
using FixedPointNumbers

@testset "ULogarithmic" begin
	@test check_grad(PlusEq(gaussian_log), (1.0, 2.0); iloss=1)
	function muleq(f, x, y, z)
		x = ULogarithmic(x)
		y = ULogarithmic(y)
		z = ULogarithmic(z)
		x *= f(y, z)
		x.log
	end
	g1 = ForwardDiff.gradient(arr->muleq(+, arr...), [7.0, 5.0, 3.0])
	x, y,z = ULogarithmic(7.0), ULogarithmic(5.0), ULogarithmic(3.0)
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
	x, y,z = ULogarithmic(2.0), ULogarithmic(5.0), ULogarithmic(3.0)
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
	res = i_exp(Fixed43(0.0), x)[1]
	gx = grad(Grad(i_exp)(Val(1), Fixed43(0.0), x)[3])
	@test res ≈ exp(3.5)
	@test gx ≈ exp(3.5)
end

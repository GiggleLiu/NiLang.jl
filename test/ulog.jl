using NiLang, NiLang.AD
using Test, Random
using ForwardDiff

@testset "basic instructions, ULogarithmic" begin
	x, y = ULogarithmic(1), ULogarithmic(2)
	@instr x *= y
	@test x == ULogarithmic(3)
	@test y == ULogarithmic(2)

	@test PlusEq(gaussian_log)(1.0, 2.0) == (1.0+log(1+exp(2.0)), 2.0)

	x, y,z = ULogarithmic(7.0), ULogarithmic(2.0), ULogarithmic(3.0)
	@instr x *= y + z
	@test check_inv(MulEq(+), (x, y, z))
	@test x.log ≈ log(exp(7.0) * (exp(2.0) + exp(3.0)))
	x, y,z = ULogarithmic(7.0), ULogarithmic(5.0), ULogarithmic(3.0)
	@instr x *= y - z
	@test x.log ≈ log(exp(7.0) * (exp(5.0) - exp(3.0)))
end

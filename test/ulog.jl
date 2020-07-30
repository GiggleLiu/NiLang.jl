using NiLang, NiLang.AD
using Test, Random
using ForwardDiff
using NiLangCore: default_constructor

@testset "basic instructions, ULogarithmic" begin
	x, y = default_constructor(ULogarithmic, 1),
		default_constructor(ULogarithmic, 2)
	@instr x *= y
	@test x == default_constructor(ULogarithmic, 3)
	@test y == default_constructor(ULogarithmic, 2)

	@test PlusEq(gaussian_log)(1.0, 2.0) == (1.0+log(1+exp(2.0)), 2.0)

	x, y, z = default_constructor(ULogarithmic, 7.0),
		default_constructor(ULogarithmic, 2.0),
		default_constructor(ULogarithmic, 3.0)
	@instr x *= y + z
	@test check_inv(MulEq(+), (x, y, z))
	@test x.log ≈ log(exp(7.0) * (exp(2.0) + exp(3.0)))
	x, y, z = default_constructor(ULogarithmic, 7.0),
		default_constructor(ULogarithmic, 5.0),
		default_constructor(ULogarithmic, 3.0)
	@instr x *= y - z
	@test x.log ≈ log(exp(7.0) * (exp(5.0) - exp(3.0)))
end

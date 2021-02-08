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

	x, y, z = default_constructor(ULogarithmic, 7.0),
		default_constructor(ULogarithmic, 5.0),
		default_constructor(ULogarithmic, 3.0)
	@instr x *= y^3.4
	@test x.log ≈ log(exp(5.0)^3.4 * exp(7.0))

	x, y, z = default_constructor(ULogarithmic, 7.0),
		default_constructor(ULogarithmic, 5.0),
		default_constructor(ULogarithmic, 3.0)
	@instr x *= 3
	@test x.log ≈ log(exp(7.0) * 3)
end


@testset "error on += and -=" begin
    @i function f(x::ULogarithmic)
        x += ULogarithmic(3.0)
    end
    
    @test_throws MethodError f(ULogarithmic(2.0))
    @test_throws MethodError (~f)(ULogarithmic(2.0))
end

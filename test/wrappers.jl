using NiLang, Test

@testset "partial" begin
    x = Partial{:im}(3+2im)
    println(x)
    @test x === Partial{:im,Complex{Int64},Int64}(3+2im)
    @test value(x) == 2
    @test chfield(x, value, 4) == Partial{:im}(3+4im)
    @test zero(x) == Partial{:im}(0.0+0.0im)
    @test (~Partial{:im})(x) == 3+2im
end

@testset "value" begin
    x = 1.0
    @test value(x) === 1.0
    @assign (x |> value) 0.2
    @test x == 0.2
end

struct NiTypeTest{T} <: IWrapper{T}
    x::T
    g::T
end
NiTypeTest(x) = NiTypeTest(x, zero(x))
@fieldview NiLang.value(invtype::NiTypeTest) = invtype.x
@fieldview gg(invtype::NiTypeTest) = invtype.g

@testset "inv type" begin
    it = NiTypeTest(0.5)
    @test eps(typeof(it)) === eps(Float64)
    @test value(it) == 0.5
    @test it ≈ NiTypeTest(0.5)
    @test it > 0.4
    @test it < NiTypeTest(0.6)
    @test it < 7
    @test 0.4 < it
    @test 7 > it
    @test chfield(it, value, 0.3) == NiTypeTest(0.3)
    it = chfield(it, Val(:g), 0.2)
    @test almost_same(NiTypeTest(0.5+1e-15), NiTypeTest(0.5))
    @test !almost_same(NiTypeTest(1.0), NiTypeTest(1))
    it = NiTypeTest(0.5)
    @test chfield(it, gg, 0.3) == NiTypeTest(0.5, 0.3)
end

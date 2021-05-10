using NiLang, Test, NiLang.AD

@testset "loaddata" begin
    @test NiLang.loaddata(GVar(0.1), 0.3) == GVar(0.3)
    @test NiLang.loaddata(Complex(GVar(0.1, 0.2), GVar(0.2)), 0.3+0.6im) == Complex(GVar(0.3), GVar(0.6))
    @test NiLang.loaddata(typeof(Complex(GVar(0.1, 0.2), GVar(0.2))), 0.3+0.6im) == Complex(GVar(0.3), GVar(0.6))
    @test NiLang.loaddata(GVar(0.2, AutoBcast{Float64,3}(zeros(3))), 0.3) == GVar(0.3, AutoBcast{Float64,3}(zeros(3)))
    @test NiLang.loaddata((GVar(0.2, AutoBcast{Float64,3}(zeros(3))), 7), (0.3, 4)) == (GVar(0.3, AutoBcast{Float64,3}(zeros(3))), 4)
    @test NiLang.loaddata(typeof((GVar(0.2, AutoBcast{Float64,3}(zeros(3))), 7)), (0.3, 4)) == (GVar(0.3, AutoBcast{Float64,3}(zeros(3))), 4)
    @test NiLang.loaddata(4, 2.0) == 2
end

@testset "push load" begin
    x = (0.3, 3.0, [1,2,3.0])
    @instr PUSH!(x)
    t = (0.0, 0.0, Float64[])
    @test x == t && typeof(x) == typeof(t)
    y = (0.0, GVar(0.0), GVar{Float64,Float64}[])
    @instr POP!(y)
    t = (0.3, GVar(3.0), GVar([1,2, 3.0]))
    @test y == t && typeof(y) == typeof(t)

    x = [0.3, 3.0, [1,2,3.0]]
    @instr PUSH!(x)
    t = []
    @test x == t && typeof(x) == typeof(t)
    y = []
    @instr POP!(y)
    t = [0.3, GVar(3.0), GVar([1,2, 3.0])]
    @test y == t && typeof(y) == typeof(t)

    x = (0.3, 3.0, [1,2,3.0])
    @instr @invcheckoff PUSH!(x)
    t = (0.0, 0.0, Float64[])
    @test x == t && typeof(x) == typeof(t)
    y = (0.0, GVar(0.0), GVar(zeros(0)))
    @instr @invcheckoff POP!(y)
    t = (0.3, GVar(3.0), GVar([1,2, 3.0]))
    @test y == t && typeof(y) == typeof(t)

    x = (0.3, 3.0, [1,2,3.0])
    @instr @invcheckoff COPYPUSH!(x)
    t = (0.3, 3.0, [1,2,3.0])
    @test x == t && typeof(x) == typeof(t)
    y = (0.3, GVar(t[2]), GVar(t[3]))
    @instr @invcheckoff COPYPOP!(y)
    t = (0.3, GVar(3.0), GVar([1,2, 3.0]))
    @test y == t && typeof(y) == typeof(t)

    x = (0.3, 3.0, [1,2,3.0])
    @instr COPYPUSH!(x)
    t = (0.3, 3.0, [1,2,3.0])
    @test x == t && typeof(x) == typeof(t)
    y = (0.3, GVar(t[2]), GVar(t[3]))
    @instr COPYPOP!(y)
    t = (0.3, GVar(3.0), GVar([1,2, 3.0]))
    @test y == t && typeof(y) == typeof(t)

    x = [0.3, 3.0, [1,2,3.0]]
    @instr COPYPUSH!(x)
    t = [0.3, 3.0, [1,2,3.0]]
    @test x == t && typeof(x) == typeof(t)
    y = [0.3, GVar(t[2]), GVar(t[3])]
    @instr COPYPOP!(y)
    t = [0.3, GVar(3.0), GVar([1,2, 3.0])]
    @test y == t && typeof(y) == typeof(t)
end
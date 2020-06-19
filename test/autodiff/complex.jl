using Test, NiLang, NiLang.AD

@testset "complex" begin
    a = 1.0+ 2im
    @instr real(a) += identity(2)
    @instr imag(a) += identity(2)
    @test a == 3.0 + 4im

    a = 1.0+ 2im
    @instr a += complex(2.0, 2.0)
    @test a == 3.0 + 4.0im
    @i function f(loss, a::Complex{T}, b) where T
        @routine begin
            c ← zero(a)
            sq ← zero(T)
            c += a * b
            sq += real(c)^2
            sq += imag(c)^2
        end
        loss += sq ^ 0.5
        ~@routine
    end
    a = 1.0 + 2.0im
    b = 2.0 + 1.0im
    loss = 0.0
    @instr f(loss, a, b)
    @test loss ≈ abs(a*b)
end

@testset "complex GVar" begin
    a = 1.0+ 2im
    @test GVar(a) == Complex(GVar(1.0), GVar(2.0))
    @test GVar(a, a) == Complex(GVar(1.0, 1.0), GVar(2.0, 2.0))
end

@i function fr(f, loss, args...; il)
    f(args...)
    loss += identity(tget(args,il).re)
end
@i function fi(f, loss, args...; il)
    f(args...)
    loss += identity(tget(args,il).im)
end
function ccheck_grad(f, args; verbose=true, iloss=1)
    check_grad(fr, (f, 0.0, args...); verbose=verbose, iloss=2, il=1) &&
     check_grad(fi, (f, 0.0, args...); verbose=verbose, iloss=2, il=1)
end

@testset "check grad" begin
    x = 1.0 - 4.0im
    y = 2.0 - 2.3im
    z = 3.0 + 1.0im
    r = 4.0
    for opm in [⊕, ⊖]
        @test check_inv(opm(complex), (1+2.0im, 2.0, 3.0); verbose=true)
        @test ccheck_grad(opm(complex), (1+2.0im, 2.0, 3.0); verbose=true, iloss=1)
        for (subop, args) in [
            (opm(identity), (x,y)), (opm(+), (x, y, z)),
            (opm(-), (x, y, z)), (opm(*), (x, y, z)),
            (opm(/), (x, y, z)), (opm(^), (x, y, r)),
            (opm(exp), (x, y)), (opm(log), (x, y))
            ]
            @test ccheck_grad(subop, args; verbose=true, iloss=1)
            r1 = subop(args...)
            r2 = [(opm == (⊕) ? Base.:+ : Base.:-)(args[1], subop.f(args[2:end]...)), args[2:end]...]
            @test all(r1 .≈ r2)
        end

        for (subop, args) in [
            (opm(angle), (r, y)), (opm(abs), (r, y)),
            (opm(abs2), (r, y))
            ]
            @show subop, args
            r1 = [subop(args...)...]
            r2 = [(opm == (⊕) ? Base.:+ : Base.:-)(args[1], subop.f(args[2:end]...)), args[2:end]...]
            @test r1 ≈ r2
            @test check_grad(subop, args; verbose=true, iloss=1)
        end
    end
    @test check_inv(NEG, (x,); verbose=true)
    @test NEG(x) == -x
    @test ccheck_grad(NEG, (x,); verbose=true, iloss=1)
end

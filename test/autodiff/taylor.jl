using NiLang, NiLang.AD
using Test
using TensorOperations
using NiLang.AD: hessian_numeric, local_hessian_numeric, local_hessian

@testset "HessianData" begin
    rings_init!(Float64)
    out! = beijingring!(6.0)
    a = beijingring!(2.0)
    b = beijingring!(3.0)

    @test nrings(Float64) == 3

    @test chfield(out!, Val(:x), 0.5).x == 0.5
    @test chfield(out!, value, 0.6).x == 0.6
    @test grad(chfield(a, grad, 0.5)) == 0.5

    @test hdata((out!, a)) == 0.0
    @instr hdata((out!, a)) ⊕ 0.5
    @instr hdata((a, out!)) ⊕ 0.5
    @test hdata((out!, a)) == 0.5
    @test hdata((a, out!)) == 0.5
end

@testset "hessian" begin
    h1 = (⊕(*)''(Loss(0.0), 2.0, 3.0); collect_hessian())
    h2 = hessian_numeric(⊕(*), (Loss(0.0), 2.0, 3.0))
    @test h1 ≈ h2

    @i function test(a,b,c)
        a += b*c
        c += b^a
        ROT(a, b, c)
    end
    h1 = (test''(Loss(0.0), 2.0, 0.5); collect_hessian())
    h2 = hessian_numeric(test, (Loss(0.0), 2.0, 0.5))
    @show h2
    @test isapprox(h1, h2, atol=1e-5)
end

function hessian_propagate(h, f, args; kwargs=())
    jac = jacobian(f, args; kwargs=kwargs)
    @tensor out[i,j,o] := jac[a,i]*h[a,b,o]*jac[b,j]
end

function hessian_propagate2(h, f, args; kwargs=())
    nargs = length(args)
    hes = zeros(nargs,nargs,nargs)
    @instr f(args...)
    for j=1:nargs
        # init rings
        rings_init!(Float64)
        largs = [beijingring!(x) for x in args]
        for i=1:nargs
            getrings(Float64)[i][1:i] .= h[1:i,i,j]
            getrings(Float64)[i][end:-1:end-i+1] .= h[i,1:i,j]
        end
        @instr (~f)(largs...)
        hes[:,:,j] .= collect_hessian()
    end
    hes
end

function rand_hes(n)
    h = randn(n,n,n)
    h + permutedims(h, (2,1,3))
end

@testset "hessian propagate" begin
    for op in [⊕(*), ⊕(/), ⊕(^), ROT, IROT]
        @show op
        h1 = local_hessian(op, (0.3, 0.4, 2.0))
        h2 = local_hessian_numeric(op, (0.3, 0.4, 2.0))
        @test h1 ≈ h2

        h = rand_hes(3)
        h1 = hessian_propagate(copy(h), op, (0.3, 0.4, 2.0))
        h2 = hessian_propagate2(copy(h), op, (0.3, 0.4, 2.0))
        @show h1 - h2
        @test h1 ≈ h2
    end

    for op in [⊕(identity), ⊕(abs), SWAP, ⊕(exp), ⊕(log), ⊕(sin), ⊕(cos)]
        @show op
        h1 = local_hessian(op, (0.3, 0.4))
        h2 = local_hessian_numeric(op, (0.3, 0.4))
        @test h1 ≈ h2

        h = rand_hes(2)
        h1 = hessian_propagate(copy(h), op, (0.3, 0.4))
        h2 = hessian_propagate2(copy(h), op, (0.3, 0.4))
        @test h1 ≈ h2
    end

    @i function f1(x)
        MULINT(x, 3)
    end
    for op in [NEG, CONJ, f1, ~f1]
        h1 = local_hessian(op, (0.3,))
        h2 = local_hessian_numeric(op, (0.3,))
        @test h1 ≈ h2

        h = rand_hes(1)
        h1 = hessian_propagate(copy(h), op, (0.3,))
        h2 = hessian_propagate2(copy(h), op, (0.3,))
        @test h1 ≈ h2
    end
end

@testset "match eltype" begin
    @test NiLang.AD.match_eltype((1, [2,3,4], [[Float64(2.0)]])) == Float64
    @test NiLang.AD.match_eltype((1, [2,3,4], [[Fixed43(2.0)]])) == Fixed43
    @test_throws ErrorException NiLang.AD.match_eltype((1, [2,3,4.0], [[Fixed43(2.0)]])) == Fixed43
end

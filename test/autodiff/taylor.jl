using NiLang, NiLang.AD
using NiLang.AD: size_paramspace
using Test
using TensorOperations

@testset "HessianData" begin
    hdata = zeros(3,3)
    gdata = [1.0, 0, 0]
    out! = HessianData(6.0, gdata, hdata, 1)
    a = HessianData(2.0, gdata, hdata, 2)
    b = HessianData(3.0, gdata, hdata, 3)

    @test size_paramspace(out!) == 3

    @test chfield(out!, Val(:x), 0.5) == HessianData(0.5, gdata, hdata, 1)
    @test chfield(out!, value, 0.6) == HessianData(0.6, gdata, hdata, 1)
    chfield(a, grad, 0.5)
    @test gdata == [1.0,0.5,0.0]
end

@testset "hessian" begin
    h1 = taylor_hessian(⊕(*), (Loss(0.0), 2.0, 3.0))
    h2 = nhessian(⊕(*), (Loss(0.0), 2.0, 3.0))
    @test h1 ≈ h2

    @i function test(a,b,c)
        a += b*c
        c += b^a
        ROT(a, b, c)
    end
    h1 = taylor_hessian(test, (Loss(0.0), 2.0, 0.5))
    h2 = nhessian(test, (Loss(0.0), 2.0, 0.5))
    @show h2
    @test isapprox(h1, h2, atol=1e-5)
end

function hessian_propagate(h, f, args; kwargs=())
    jac = jacobian(f, args; kwargs=kwargs)
    @tensor out[i,j,o] := jac[i,a]*h[a,b,o]*jac[j,b]
end

function hessian_propagate2(h, f, args; kwargs=())
    nargs = length(args)
    hes = zeros(nargs,nargs,nargs)
    @instr f(args...)
    for j=1:nargs
        gdata = zeros(nargs)
        hdata = h[:,:,j]
        largs = [HessianData(arg, gdata, hdata, i) for (i, arg) in enumerate(args)]
        @instr (~f)(largs...)
        hes[:,:,j] .= hdata
    end
    hes
end

function rand_hes(n)
    h = randn(n,n,n)
    h + permutedims(h, (2,1,3))
end

@testset "hessian propagate" begin
    for op in [⊕(*), ⊕(/), ⊕(^), ROT]
        @show op
        h1 = local_hessian(op, (0.3, 0.4, 2.0))
        h2 = local_nhessian(op, (0.3, 0.4, 2.0))
        @test h1 ≈ h2

        h = rand_hes(3)
        h1 = hessian_propagate(copy(h), op, (0.3, 0.4, 2.0))
        h2 = hessian_propagate2(copy(h), op, (0.3, 0.4, 2.0))
        @test h1 ≈ h2
    end

    for op in [⊕(identity), SWAP]
        h1 = local_hessian(op, (0.3, 0.4))
        h2 = local_nhessian(op, (0.3, 0.4))
        @test h1 ≈ h2

        h = rand_hes(2)
        h1 = hessian_propagate(copy(h), op, (0.3, 0.4))
        h2 = hessian_propagate2(copy(h), op, (0.3, 0.4))
        @test h1 ≈ h2
    end

    for op in [NEG, CONJ]
        h1 = local_hessian(op, (0.3,))
        h2 = local_nhessian(op, (0.3,))
        @test h1 ≈ h2

        h = rand_hes(1)
        h1 = hessian_propagate(copy(h), op, (0.3,))
        h2 = hessian_propagate2(copy(h), op, (0.3,))
        @test h1 ≈ h2
    end
end

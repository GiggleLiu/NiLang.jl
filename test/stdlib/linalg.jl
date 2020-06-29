using Test
using NiLang, NiLang.AD
using LinearAlgebra
using Random

@testset "inv" begin
    Random.seed!(2)

    @test check_inv(i_inv!, (randn(3, 3), randn(3, 3)))
    @test check_inv(⊕(det), (0.3, randn(3, 3)))
    @test check_inv(⊕(logdet), (0.3, randn(3, 3)))
    @test check_grad(⊕(det), (0.3, randn(3, 3)), iloss=1)
    @test check_grad(⊕(logdet), (0.3, randn(3, 3)), iloss=1)

    @i function loss(out!, y, A)
        i_inv!(y, A)
        out! += y[1,1]
    end
    @test check_grad(loss, (0.0, randn(3, 3), randn(3, 3)); iloss=1)
end

@testset "affine" begin
    Random.seed!(2)
    A = randn(5, 5)
    b = randn(5)
    x = randn(5)
    y! = zeros(5)
    @test i_affine!(y!, A, b, x)[1] ≈ A*x + b
end

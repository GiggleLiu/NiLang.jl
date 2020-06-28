using Test, LightBayesian
using NiLang, NiLang.AD
using LinearAlgebra
using Random

@testset "inv" begin
    Random.seed!(2)

    @test check_inv(i_inv, (randn(3, 3), randn(3, 3)))
    @test check_inv(⊕(det), (0.3, randn(3, 3)))
    @test check_inv(⊕(logdet), (0.3, randn(3, 3)))
    @test check_grad(⊕(det), (0.3, randn(3, 3)), iloss=1)
    @test check_grad(⊕(logdet), (0.3, randn(3, 3)), iloss=1)

    @i function loss(out!, y, A)
        i_inv(y, A)
        out! += identity(y[1,1])
    end
    @test check_grad(loss, (0.0, randn(3, 3), randn(3, 3)); iloss=1)
end





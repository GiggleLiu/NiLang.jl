# bijectivity check
using Test
include("nice.jl")

@testset "nice" begin
    num_vars = 4
    model = random_nice_network(num_vars, 10, 3)
    z = randn(num_vars)
    x, _ = nice_network!(z, model)
    z_infer, _ = (~nice_network!)(x, model)
    @test z_infer ≈ z
    newparams = randn(nparameters(model))
    dispatch_params!(model, newparams)
    @test collect_params(model) ≈ newparams
    @test check_inv(logp!, (0.0, x, model))
end

@testset "nice logp" begin
    z1 = [0.5, 0.2]
    z2 = [-0.5, 1.2]
    model = random_nice_network(2, 10, 4)
    x1 = nice_network!(copy(z1), model)[1]
    x2 = nice_network!(copy(z2), model)[1]
    p1 = logp!(0.0, copy(x1), model)[1]
    p2 = logp!(0.0, copy(x2), model)[1]
    pz1 = exp(-sum(abs2, z1)/2)
    pz2 = exp(-sum(abs2, z2)/2)
    @test exp(p1 - p2) ≈ pz1/pz2
    @test nice_nll!(0.0, 0.0, hcat(x1, x2), model)[1] ≈ -log(pz1 * pz2)/2

    xs = hcat(x1, x2)
    gmodel = Grad(nice_nll!)(Val(1), 0.0, 0.0, copy(xs), model)[end]

    for i=1:10, j=1:4
        model[j].W2[i] -= 1e-4
        a = nice_nll!(0.0, 0.0, copy(xs), model)[1]
        model[j].W2[i] += 2e-4
        b = nice_nll!(0.0, 0.0, copy(xs), model)[1]
        model[j].W2[i] -= 1e-4
        ng = (b-a)/2e-4
        @test gmodel[j].W2[i].g ≈ ng
    end

    for i=1:10, j=1:4
        model[j].W1[i] -= 1e-4
        a = nice_nll!(0.0, 0.0, copy(xs), model)[1]
        model[j].W1[i] += 2e-4
        b = nice_nll!(0.0, 0.0, copy(xs), model)[1]
        model[j].W1[i] -= 1e-4
        ng = (b-a)/2e-4
        @test gmodel[j].W1[i].g ≈ ng
    end
end



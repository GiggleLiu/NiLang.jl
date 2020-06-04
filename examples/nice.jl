using NiLang, NiLang.AD
using LinearAlgebra
using DelimitedFiles
using Plots

include("Adam.jl")

struct NiceLayer{T}
    W1::Matrix{T}
    b1::Vector{T}
    W2::Matrix{T}
    b2::Vector{T}
end
NiLang.AD.GVar(x::NiceLayer) = NiceLayer(GVar(x.W1), GVar(x.b1), GVar(x.W2), GVar(x.b2))

function random_nice_network(nparams::Int, nhidden::Int, nlayer::Int; scale=0.1)
    random_nice_network(Float64, nparams, nhidden, nlayer; scale=scale)
end

function random_nice_network(::Type{T}, nparams::Int, nhidden::Int, nlayer::Int; scale=0.1) where T
    nin = nparams÷2
    scale = T(scale)
    NiceLayer{T}[NiceLayer(randn(T, nhidden, nin)*scale, randn(T, nhidden)*scale,
            randn(T, nin, nhidden)*scale, randn(T, nin)*scale) for _ = 1:nlayer]
end

const NiceNetwork{T} = Vector{NiceLayer{T}}

nparameters(n::NiceLayer) = length(n.W1) + length(n.b1) + length(n.W2) + length(n.b2)
nparameters(n::NiceNetwork) = sum(nparameters, n)

function collect_params(n::NiceNetwork{T}) where T
    out = zeros(T, nparameters(n))
    k = 0
    for layer in n
        np = nparameters(layer)
        collect_params!(view(out, k+1:k+np), layer)
        k += np
    end
    return out
end

function dispatch_params!(network::NiceNetwork, out)
    k = 0
    for layer in network
        np = nparameters(layer)
        dispatch_params!(layer, view(out, k+1:k+np))
        k += np
    end
    return network
end

function collect_params!(out, layer::NiceLayer)
    a, b, c, d = length(layer.W1), length(layer.b1), length(layer.W2), length(layer.b2)
    out[1:a] .= vec(layer.W1)
    out[a+1:a+b] .= layer.b1
    out[a+b+1:a+b+c] .= vec(layer.W2)
    out[a+b+c+1:end] .= layer.b2
    return out
end

function dispatch_params!(layer::NiceLayer, out)
    a, b, c, d = length(layer.W1), length(layer.b1), length(layer.W2), length(layer.b2)
    vec(layer.W1) .= out[1:a]
    layer.b1 .= out[a+1:a+b]
    vec(layer.W2) .= out[a+b+1:a+b+c]
    layer.b2 .= out[a+b+c+1:end]
    return layer
end

@i function relu(y!, x::T) where T
    if (x > 0, ~)
        y! += identity(x)
    end
end

@i function affine!(y!, W, b, x)
    @safe @assert size(W) == (length(y!), length(x)) && length(b) == length(y!)
    @invcheckoff for j=1:size(W, 2)
        for i=1:size(W, 1)
            @inbounds y![i] += W[i,j]*x[j]
        end
    end
    @invcheckoff for i=1:size(W, 1)
        @inbounds y![i] += identity(b[i])
    end
end

@i function nice_layer!(x::AbstractVector{T}, layer::NiceLayer{T},
                y!::AbstractVector{T}) where T
    @routine @invcheckoff begin
        y1 ← zeros(T, size(layer.W1, 1))
        y1a ← zero(y1)
        affine!(y1, layer.W1, layer.b1, x)
        for i=1:length(y1)
            if (y1[i] > 0, ~)
                @inbounds y1a[i] += identity(y1[i])
            end
        end
    end
    affine!(y!, layer.W2, layer.b2, y1a)
    ~@routine
end

@i function nice_network!(x!::AbstractVector{T}, network::NiceNetwork{T}) where T
    @invcheckoff for i=1:length(network)
        np ← length(x!)
        if (i%2==0, ~)
            @inbounds nice_layer!(view(x!,np÷2+1:np), network[i], view(x!,1:np÷2))
        else
            @inbounds nice_layer!(view(x!,1:np÷2), network[i], view(x!,np÷2+1:np))
        end
    end
end

@i function logp!(out!::T, x!::AbstractVector{T}, network::NiceNetwork{T}) where T
    (~nice_network!)(x!, network)
    @invcheckoff for i = 1:length(x!)
        @routine begin
            xsq ← zero(T)
            @inbounds xsq += x![i]^2
        end
        out! -= 0.5 * xsq
        ~@routine
    end
end

@i function nice_nll!(out!::T, cum!::T, xs!::Matrix{T}, network::NiceNetwork{T}) where T
    @invcheckoff for i=1:size(xs!, 2)
        @inbounds logp!(cum!, view(xs!,:,i), network)
    end
    out! -= cum!/size(xs!, 2)
end

# bijectivity check
using Test
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

function train(x_data, model; num_epochs = 800)
    num_vars = size(x_data, 1)
    params = collect_params(model)
    optimizer = Adam(; lr=0.01)
    for epoch = 1:num_epochs
        loss, a, b, c = nice_nll!(0.0, 0.0, copy(x_data), model)
        if epoch % 50 == 1
            println("epoch = $epoch, loss = $loss")
            display(showmodel(x_data, model))
        end
        _, _, _, gmodel = (~nice_nll!)(GVar(loss, 1.0), GVar(a), GVar(b), GVar(c))
        g = grad.(collect_params(gmodel))
        update!(params, grad.(collect_params(gmodel)), optimizer)
        dispatch_params!(model, params)
    end
    return model
end

function showmodel(x_data, model; nsamples=2000)
    scatter(x_data[1,1:nsamples], x_data[2,1:nsamples]; xlims=(-5,5), ylims=(-5,5))
    zs = randn(2, nsamples)
    for i=1:nsamples
        nice_network!(view(zs, :, i), model)
    end
    scatter!(zs[1,:], zs[2,:])
end

import Random; Random.seed!(22)
x_data = Matrix(readdlm(joinpath(@__DIR__, "train.dat"))')
model = random_nice_network(Float64, size(x_data, 1), 10, 4; scale=1)
@time model = train(x_data, model; num_epochs=800)

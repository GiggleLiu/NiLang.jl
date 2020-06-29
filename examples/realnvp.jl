# # RealNVP network
# For the definition of this network and concepts of normalizing flow,
# please refer this realnvp blog: https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html,
# and the pytorch notebook: https://github.com/GiggleLiu/marburg/blob/master/solutions/realnvp.ipynb

using NiLang, NiLang.AD
using LinearAlgebra
using DelimitedFiles
using Plots

# `include` the optimizer, you can find it under the `Adam.jl` file in the `examples/` folder.
include("Adam.jl")


# ## Model definition
# First, define the single layer transformation and its behavior under `GVar` - the gradient wrapper.
struct RealNVPLayer{T}
    ## transform network
    W1::Matrix{T}
    b1::Vector{T}
    W2::Matrix{T}
    b2::Vector{T}
    y1::Vector{T}
    y1a::Vector{T}

    ## scaling network
    sW1::Matrix{T}
    sb1::Vector{T}
    sW2::Matrix{T}
    sb2::Vector{T}
    sy1::Vector{T}
    sy1a::Vector{T}
end

"""collect parameters in the `layer` into a vector `out`."""
function collect_params!(out, layer::RealNVPLayer)
    k=0
    for field in [:W1, :b1, :W2, :b2, :sW1, :sb1, :sW2, :sb2]
        v = getfield(layer, field)
        nv = length(v)
        out[k+1:k+nv] .= vec(v)
        k += nv
    end
    return out
end

"""dispatch vectorized parameters `out` into the `layer`."""
function dispatch_params!(layer::RealNVPLayer, out)
    k=0
    for field in [:W1, :b1, :W2, :b2, :sW1, :sb1, :sW2, :sb2]
        v = getfield(layer, field)
        nv = length(v)
        vec(v) .= out[k+1:k+nv]
        k += nv
    end
    return out
end

function nparameters(n::RealNVPLayer)
    sum(x->length(getfield(n, x)), [:W1, :b1, :W2, :b2, :sW1, :sb1, :sW2, :sb2])
end

# Then, we define `network` and how to access the parameters.
const RealNVP{T} = Vector{RealNVPLayer{T}}

nparameters(n::RealNVP) = sum(nparameters, n)

function collect_params(n::RealNVP{T}) where T
    out = zeros(T, nparameters(n))
    k = 0
    for layer in n
        np = nparameters(layer)
        collect_params!(view(out, k+1:k+np), layer)
        k += np
    end
    return out
end

function dispatch_params!(network::RealNVP, out)
    k = 0
    for layer in network
        np = nparameters(layer)
        dispatch_params!(layer, view(out, k+1:k+np))
        k += np
    end
    return network
end

function random_realnvp(nparams::Int, nhidden::Int, nhidden_s::Int, nlayer::Int; scale=0.1)
    random_realnvp(Float64, nparams, nhidden, nhidden_s::Int, nlayer; scale=scale)
end

function random_realnvp(::Type{T}, nparams::Int, nhidden::Int, nhidden_s::Int, nlayer::Int; scale=0.1) where T
    nin = nparams÷2
    scale = T(scale)
    y1 = zeros(T, nhidden)
    sy1 = zeros(T, nhidden_s)
    RealNVPLayer{T}[RealNVPLayer(
            randn(T, nhidden, nin)*scale, randn(T, nhidden)*scale,
            randn(T, nin, nhidden)*scale, randn(T, nin)*scale, y1, zero(y1),
            randn(T, nhidden_s, nin)*scale, randn(T, nhidden_s)*scale,
            randn(T, nin, nhidden_s)*scale, randn(T, nin)*scale, sy1, zero(sy1),
            ) for _ = 1:nlayer]
end


# ## Loss function
#
# In each layer, we use the information in `x` to update `y!`.
# During computing, we use to vector type ancillas `y1` and `y1a`,
# both of them can be uncomputed at the end of the function.

@i function onelayer!(x::AbstractVector{T}, layer::RealNVPLayer{T},
                y!::AbstractVector{T}, logjacobian!::T; islast) where T
    @routine @invcheckoff begin
        ## scale network
        scale ← zero(y!)
        ytemp2 ← zero(y!)
        i_affine!(layer.sy1, layer.sW1, layer.sb1, x)
        @inbounds for i=1:length(layer.sy1)
            if (layer.sy1[i] > 0, ~)
                layer.sy1a[i] += layer.sy1[i]
            end
        end
        i_affine!(scale, layer.sW2, layer.sb2, layer.sy1a)

        ## transform network
        i_affine!(layer.y1, layer.W1, layer.b1, x)
        ## relu
        @inbounds for i=1:length(layer.y1)
            if (layer.y1[i] > 0, ~)
                layer.y1a[i] += layer.y1[i]
            end
        end
    end
    ## inplace multiply exp of scale! -- dangerous
    @inbounds @invcheckoff for i=1:length(scale)
        @routine begin
            expscale ← zero(T)
            tanhscale ← zero(T)
            if (islast, ~)
                tanhscale += tanh(scale[i])
            else
                tanhscale += scale[i]
            end
            expscale += exp(tanhscale)
        end
        logjacobian! += tanhscale
        ## inplace multiply!!!
        temp ← zero(T)
        temp += y![i] * expscale
        SWAP(temp, y![i])
        temp -= y![i] / expscale
        temp → zero(T)
        ~@routine
    end

    ## affine the transform layer
    i_affine!(y!, layer.W2, layer.b2, layer.y1a)
    ~@routine
    ## clean up accumulated rounding error, since this memory is reused.
    @safe layer.y1 .= zero(T)
    @safe layer.sy1 .= zero(T)
end

# A realnvp network always transforms inputs reversibly.
# We update one half of `x!` a time, so that input and output memory space do not clash.
@i function realnvp!(x!::AbstractVector{T}, network::RealNVP{T}, logjacobian!) where T
    @invcheckoff for i=1:length(network)
        np ← length(x!)
        if (i%2==0, ~)
            @inbounds onelayer!(view(x!,np÷2+1:np), network[i], view(x!,1:np÷2), logjacobian!; islast=i==length(network))
        else
            @inbounds onelayer!(view(x!,1:np÷2), network[i], view(x!,np÷2+1:np), logjacobian!; islast=i==length(network))
        end
    end
end

# How to obtain the log-probability of a data.

@i function logp!(out!::T, x!::AbstractVector{T}, network::RealNVP{T}) where T
    (~realnvp!)(x!, network, out!)
    @invcheckoff for i = 1:length(x!)
        @routine begin
            xsq ← zero(T)
            @inbounds xsq += x![i]^2
        end
        out! -= 0.5 * xsq
        ~@routine
    end
end

# The negative-log-likelihood loss function

@i function nll_loss!(out!::T, cum!::T, xs!::Matrix{T}, network::RealNVP{T}) where T
    @invcheckoff for i=1:size(xs!, 2)
        @inbounds logp!(cum!, view(xs!,:,i), network)
    end
    out! -= cum!/size(xs!, 2)
end

# ## Training

function train(x_data, model; num_epochs = 800)
    num_vars = size(x_data, 1)
    params = collect_params(model)
    optimizer = Adam(; lr=0.01)
    for epoch = 1:num_epochs
        loss, a, b, c = nll_loss!(0.0, 0.0, copy(x_data), model)
        if epoch % 50 == 1
            println("epoch = $epoch, loss = $loss")
            display(showmodel(x_data, model))
        end
        _, _, _, gmodel = (~nll_loss!)(GVar(loss, 1.0), GVar(a), GVar(b), GVar(c))
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
        realnvp!(view(zs, :, i), model, 0.0)
    end
    scatter!(zs[1,:], zs[2,:])
end

# you can find the training data in `examples/` folder
x_data = Matrix(readdlm(joinpath(@__DIR__, "train.dat"))')

import Random; Random.seed!(22)
model = random_realnvp(Float64, size(x_data, 1), 10, 10, 4; scale=0.1)

# Before training, the distribution looks like
# ![before](../../asset/nice_before.png)
model = train(x_data, model; num_epochs=800)

# After training, the distribution looks like
# ![before](../../asset/realnvp_after.png)

export softmax_cross_entropy

# TODO: error on `scalar .+ vector`
# TODO: allow docstring
@i function softmax_cross_entropy(x, p, imax, xmax, Z, out::T) where T
    @anc logZ = zero(T)
    @anc yi = zero(T)
    # subtract maximum
    imax += argmax(x)  # trade off space of xmax to time
    xmax ⊕ x[imax]
    # accumulate exp(x) to Z, and finally get logZ
    for i=1:length(x)
        x[i] ⊖ xmax
        Z += exp(x[i])
    end
    logZ += log(Z)
    for i=1:length(x)
        yi ⊕ logZ
        yi ⊖ x[i]
        out += yi * p[i]
        yi ⊕ x[i]
        yi ⊖ logZ
    end
    logZ -= log(Z)
end

function _sce(x::AbstractArray{T,N}, p) where {T,N}
    x = x .- maximum(x; dims=N)  # avoid data overflow
    rho = exp.(x)
    Z = sum(rho; dims=N)
    return dropdims(sum((log.(Z) .- x) .* p; dims=N), dims=N)
end

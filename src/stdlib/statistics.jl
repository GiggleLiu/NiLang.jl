export i_mean_sum, i_var_mean_sum, i_normal_logpdf

"""
    i_mean_sum(out!, sum!, x)

get the `mean` and `sum` of `x`.
"""
@i function i_mean_sum(out!, sum!, x)
    for i=1:length(x)
        sum! += x[i]
    end
    out! += sum!/length(x)
end

"""
    i_var_mean_sum(var!, varsum!, mean!, sum!, sqv)

Compute the variance, the accumulated variance, mean and sum.
"""
@i function i_var_mean_sum(var!, varsum!, mean!, sum!, v::AbstractVector{T}) where T
    i_mean_sum(mean!, sum!, v)
    for i=1:length(v)
        v[i] -= mean!
        varsum! += v[i] ^ 2
        v[i] += mean!
    end
    var! += varsum! / (length(v)-1)
end

"""
    i_normal_logpdf(out, x, μ, σ)

get the pdf of `Normal(μ, σ)` at point `x`.
"""
@i function i_normal_logpdf(out, x::T, μ, σ) where T
    @zeros T anc1 anc2 anc3

    @routine begin
        anc1 += x
        anc1 -= μ
        anc2 += anc1 / σ  # (x- μ)/σ
        anc3 += anc2^2 # (x-μ)^2/σ^2
    end

    out -= anc3 * 0.5 # -(x-μ)^2/2σ^2
    out -= log(σ) # -(x-μ)^2/2σ^2 - log(σ)
    out -= log(2π)/2 # -(x-μ)^2/2σ^2 - log(σ) - log(2π)/2

    ~@routine
end

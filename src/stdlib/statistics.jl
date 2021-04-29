export i_mean_sum, i_mean, i_sum, i_var_mean_sum, i_normal_logpdf, i_cor_cov
export VarianceInfo

"""
    i_mean_sum(mean_val, sum_val, x)

Get the `mean` and `sum` of `x` in one routine.

```jldoctest; setup=:(using NiLang)
x = rand(64, 64)
m, s, x = i_mean_sum(0.0, 0.0, x)
(m, s) .≈ (mean(x), sum(x))

# output
(true, true)
```

See also: [`i_mean`](@ref), [`i_sum`](@ref)
"""
@i function i_mean_sum(mean_val, sum_val, x)
    @inbounds @simd for i in 1:length(x)
        sum_val += x[i]
    end
    mean_val += sum_val/(@const length(x))
end

"""
    i_sum(sum_val, x)

Get the `sum` of `x`.

```jldoctest; setup=:(using NiLang)
x = rand(64, 64)
i_sum(0.0, x)[1] ≈ sum(x)

# output
true
```

!!! tip
    If you need both `mean` and `sum` of `x`, then you can use a more
    performant version [`i_mean_sum`](@ref).
"""
@i function i_sum(sum_val, x)
    @inbounds @simd for i in 1:length(x)
        sum_val += x[i]
    end
end

"""
    i_mean(mean_val, x)

Get the `mean` of `x`.

```jldoctest; setup=:(using NiLang)
x = rand(64, 64)
i_mean(0.0, x)[1] ≈ Statistics.mean(x)

# output
true
```

!!! tip
    If you need both `mean` and `sum` of `x`, then you can use a more
    performant version [`i_mean_sum`](@ref).
"""
@i function i_mean(mean_val, x::AbstractArray)
    @routine @invcheckoff begin
        sum_val ← zero(eltype(x))
        @inbounds @simd for i in 1:length(x)
            sum_val += x[i]
        end
    end
    mean_val += sum_val/(@const length(x))
    ~@routine
end

struct VarianceInfo{T}
    variance::T
    variance_accumulated::T
    mean::T
    sum::T
end

function VarianceInfo(::Type{T}) where T
    VarianceInfo(zero(T), zero(T), zero(T), zero(T))
end

"""
    i_var_mean_sum(varinfo, sqv)
    i_var_mean_sum(var!, varsum!, mean!, sum!, v)

Compute the variance, the accumulated variance, mean and sum.
`varinfo` is the `VarianceInfo` object to store outputs.
"""
@i function i_var_mean_sum(varinfo::VarianceInfo{T}, v::AbstractVector{T}) where T
    i_var_mean_sum(varinfo.variance, varinfo.variance_accumulated, varinfo.mean, varinfo.sum, v)
end

@i function i_var_mean_sum(var!, varsum!, mean!, sum!, v::AbstractVector{T}) where T
    i_mean_sum(mean!, sum!, v)
     for i=1:length(v)
        @routine @invcheckoff begin
            x ← zero(T)
            x += v[i] - mean!
        end
        varsum! += x ^ 2
        ~@routine
     end
    var! += varsum! / (@const length(v)-1)
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

"""
     i_cor_cov(rho!,cov!,a,b)

get Pearson correlation and covariance of two vectors `a` and `b` 

"""

@i function i_cor_cov(rho!::T, cov!::T, a::AbstractVector{T}, b::AbstractVector{T}) where T
    @safe @assert length(a) == length(b)
    @routine  @invcheckoff begin
        @zeros T std1 std2
        info1 ← _zero(VarianceInfo{T})
        i_var_mean_sum(info1, a)
        std1 += sqrt(info1.variance)
        info2 ← _zero(VarianceInfo{T})
        i_var_mean_sum(info2, b)
        std2 += sqrt(info2.variance)
        @zeros T anc5 anc6 anc7
        @inbounds for i=1:length(b)
            @routine begin
                @zeros T anc3 anc4
                anc3 += a[i] - info1.mean
                anc4 += b[i] - info2.mean
            end
            anc5 += anc3 * anc4
            ~@routine
        end
        anc6 += std1 * std2
        anc7 += anc6 * (@const length(b)-1)
    end
    cov! += anc5 / (@const length(b)-1)
    rho! += anc5 / anc7 
    ~@routine
end

"""the mean value"""
@i function i_mean(out!, sum!, x)
    for i=1:length(x)
        sum! += identity(x[i])
    end
    out! += sum!/length(x)
end

"""
    var_and_mean_sq(var!, mean!, sqv)

the variance and mean value.
"""
@i function var_and_mean(var!, mean!, sum!, v::AbstractVector{T}) where T
    i_mean(mean!, sum!, v)
    for i=1:length(v)
        v[i] -= identity(mean!)
        var! += v[i] ^ 2
        v[i] += identity(mean!)
    end
    divint(var!, length(v)-1)
end

@i function normal_logpdf(out, x::T, μ, σ) where T
    anc1 ← zero(T)
    anc2 ← zero(T)
    anc3 ← zero(T)

    @routine begin
        anc1 ⊕ x
        anc1 ⊖ μ
        anc2 += anc1 / σ  # (x- μ)/σ
        anc3 += anc2 * anc2 # (x-μ)^2/σ^2
    end

    out -= anc3 * 0.5 # -(x-μ)^2/2σ^2
    out -= log(σ) # -(x-μ)^2/2σ^2 - log(σ)
    out -= log(2π)/2 # -(x-μ)^2/2σ^2 - log(σ) - log(2π)/2

    ~@routine
end

@i function normal_logpdf2d(out::T, x, μ, σ) where T
    temp1 ← zero(T)
    temp2 ← zero(T)
    normal_logpdf(temp1, x[1], μ[1], σ)
    normal_logpdf(out, x[2], μ[2], σ)
    out ⊕ temp1
    (~normal_logpdf)(temp1, x[1], μ[1], σ)
end




# # Pyramid example
#
# This is the Pyramid example in the book "Evaluate Derivatives", Sec. 3.5.

using NiLang, NiLang.AD

@i function pyramid!(y!, v!, x::AbstractVector{T}) where T
    @safe @assert size(v!,2) == size(v!,1) == length(x)
    @invcheckoff @inbounds for j=1:length(x)
        v![1,j] += x[j]
    end
    @invcheckoff @inbounds for i=1:size(v!,1)-1
        for j=1:size(v!,2)-i
            @routine begin
                @zeros T c s
                c += cos(v![i,j+1])
                s += sin(v![i,j])
            end
            v![i+1,j] += c * s
            ~@routine
        end
    end
    y! += v![end,1]
end

x = randn(20)
pyramid!(0.0, zeros(20, 20), x)

# Let' compare our implementation with those in the book
# ![](pyramid-benchmark.png)
using BenchmarkTools
@benchmark gradient(Val(1), pyramid!, (0.0, zeros(20, 20), $x))

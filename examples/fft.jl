# https://rosettacode.org/wiki/Fast_Fourier_transform#Fortran
# In place Cooley-Tukey FFT
function fft!(x::AbstractVector{T}) where T
    N = length(x)
    @inbounds if N <= 1
        return x
    elseif N == 2
        t =  x[2]
        oi = x[1]
        x[1]     = oi + t
        x[2]     = oi - t
        return x
    end
 
    # divide
    odd  = x[1:2:N]
    even = x[2:2:N]
 
    # conquer
    fft!(odd)
    fft!(even)
 
    # combine
    @inbounds for i=1:N÷2
       t = exp(T(-2im*π*(i-1)/N)) * even[i]
       oi = odd[i]
       x[i]     = oi + t
       x[i+N÷2] = oi - t
    end
    return x
end

using NiLang
@i function i_fft!(x::AbstractVector{T}) where T
    @invcheckoff N ← length(x)
    @safe @assert N%2 == 0
    @invcheckoff @inbounds if N <= 1
    elseif N == 2
        HADAMARD(x[1].re, x[2].re)
        HADAMARD(x[1].im, x[2].im)
    else
        # devide and conquer
        i_fft!(x[1:2:N])
        i_fft!(x[2:2:N])

        x2 ← zeros(T, N)
        for i=1:N÷2
            x2[i] += x[2i-1]
            x2[i+N÷2] += x[2i]
        end
        for i=1:N
            SWAP(x[i], x2[i])
        end
        for i=1:N÷2
            x2[2i-1] -= x[i]
            x2[2i] -= x[i+N÷2]
        end
        # combine
        for i=1:N÷2
            θ ← -2*π*(i-1)/N
            ROT(x[i+N÷2].re, x[i+N÷2].im, θ)
            HADAMARD(x[i].re, x[i+N÷2].re)
            HADAMARD(x[i].im, x[i+N÷2].im)
        end
    end
end

using Test, FFTW
@testset "fft" begin
    x = randn(ComplexF64, 64)
    @test fft!(copy(x)) ≈ FFTW.fft(x)
    @test i_fft!(copy(x)) .* sqrt(length(x)) ≈ FFTW.fft(x)
end
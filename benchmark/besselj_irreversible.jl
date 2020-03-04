using Zygote
using ForwardDiff
using BenchmarkTools

function besselj(ν, z; atol=1e-8)
    k = 0
    s = (z/2)^ν / factorial(ν)
    out = s
    while abs(s) > atol
        k += 1
        s *= (-1) / k / (k+ν) * (z/2)^2
        out += s
    end
    out
end

function grad_besselj_manual(ν, z; atol=1e-8)
    (besselj(ν-1, z; atol=atol) - besselj(ν+1, z); atol=atol)/2
end

println("Benchmarking Julia")
display(@benchmark besselj(2, 1.0))
println("Benchmarking Manual")
display(@benchmark grad_besselj_manual(2, 1.0))
println("Benchmarking Zygote")
display(@benchmark Zygote.gradient(besselj, 2, 1.0))
println("Benchmarking ForwardDiff")
display(@benchmark ForwardDiff.derivative(x->besselj(2, x), 1.0))

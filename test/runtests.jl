using NiLang
using Test

@testset "instructs.jl" begin
    include("instructs.jl")
end

@testset "autodiff" begin
    include("autodiff/autodiff.jl")
end

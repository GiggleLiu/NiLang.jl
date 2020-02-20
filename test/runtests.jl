using NiLang
using Test

@testset "instructs.jl" begin
    include("instructs.jl")
end

@testset "autobcast.jl" begin
    include("autobcast.jl")
end

@testset "autodiff" begin
    include("autodiff/autodiff.jl")
end

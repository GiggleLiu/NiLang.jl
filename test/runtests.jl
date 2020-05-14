using NiLang
using Test

@testset "utils.jl" begin
    include("utils.jl")
end

@testset "instructs.jl" begin
    include("instructs.jl")
end

@testset "autobcast.jl" begin
    include("autobcast.jl")
end

@testset "autodiff" begin
    include("autodiff/autodiff.jl")
end

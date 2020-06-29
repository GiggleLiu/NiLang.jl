using NiLang
using Test

@testset "vars.jl" begin
    include("vars.jl")
end

@testset "utils.jl" begin
    include("utils.jl")
end

@testset "instructs.jl" begin
    include("instructs.jl")
end

@testset "stack.jl" begin
    include("stack.jl")
end

@testset "autobcast.jl" begin
    include("autobcast.jl")
end

@testset "autodiff" begin
    include("autodiff/autodiff.jl")
end

@testset "stdlib" begin
    include("stdlib/stdlib.jl")
end

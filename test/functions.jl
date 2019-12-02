using NiLang
using Test

function naive_umm!(x, params)
    N = size(x, 1)
    k = 0
    for j=1:N
        for i=N-1:-1:j
            k += 1
            a, b = rot(x[i],x[i+1],params[k])
            x[i], x[i+1] = a, b
        end
    end
end

function inv_naive_umm!(x, params)
    N = size(x, 1)
    k = N*(N-1) ÷ 2
    for j=N:-1:1
        for i=j:N-1
            a, b = rot(x[i],x[i+1],-params[k])
            x[i], x[i+1] = a, b
            k -= 1
        end
    end
end

@testset "naive unitary" begin
    x = randn(200)
    params = randn(100*199).*2π

    x0 = copy(x)
    params0 = copy(params)
    naive_umm!(x, params)
    inv_naive_umm!(x, params)
    @test params ≈ params0
    @test x ≈ x0
end

@testset "unitary" begin
    x = randn(20)
    params = randn(10*19) * 2π

    x0 = copy(x)
    params0 = copy(params)
    Nx = length(x)
    @instr umm!(x, params, Nx, Nx)
    x1 = copy(x0)
    params1 = copy(params0)
    naive_umm!(x1, params1)
    @test params ≈ params1
    @test x ≈ x1
    @instr (~umm!)(x, params, Nx, Nx)
    @test params ≈ params0
    @test x ≈ x0
    @test check_inv(umm!, (x, params, Nx, Nx))
end

@testset "imapfoldl" begin
    x = randn(20)
    out = 0.0
    @test_throws InvertibilityError MapFoldl(exp, +)
    @instr MapFoldl(exp, ⊕)(out, x)
    @test out ≈ sum(exp.(x))
    @test check_inv(MapFoldl(exp, ⊕), (out, x))
end

@testset "inorm2, dot" begin
    out = 0.0
    vec = [1.0, 2.0, 3.0]
    vec2 = [1.0, 2.0, 5.0]
    @instr inorm2(out, vec)
    @test out ≈ norm(vec)^2
    @test check_inv(inorm2, (out, vec))

    out = 0.0
    vec = [1.0, 2.0, 3.0]
    vec2 = [1.0, 2.0, 5.0]
    @test check_grad(inorm2, (Loss(out), vec); verbose=true)

    out = 0.0
    @instr idot(out, vec, vec2)
    @test out ≈ dot(vec, vec2)
    @test check_inv(idot, (out, vec, vec2))

    @test check_grad(idot, (Loss(0.0), vec, vec2); verbose=true)

    m = randn(4,4)
    n = randn(4,4)
    out = 0.0
    @instr idot(out, m[:,2], n[:,4])
    @test out ≈ dot(m[:,2], n[:,4])
    @test check_inv(idot, (out, m[:,2], n[:,4]))

    @test check_grad(idot, (Loss(0.0), vec, vec2); verbose=true)
end

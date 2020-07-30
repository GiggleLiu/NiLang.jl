using NiLang, NiLang.AD
using Test

@testset "check grad" begin
    for opm in [⊕, ⊖]
        @test check_grad(opm(identity), (1.0, 2.0); verbose=true, iloss=1)
        @test check_grad(opm(*), (1.0, 2.0, 2.0); verbose=true, iloss=1)
        @test check_grad(opm(+), (1.0, 2.0, 2.0); verbose=true, iloss=1)
        @test check_grad(opm(-), (1.0, 2.0, 2.0); verbose=true, iloss=1)
        @test check_grad(opm(^), (1.0, 2.0, 2); verbose=true, iloss=1)
        @test check_grad(opm(^), (1.0, 2.0, 2.0); verbose=true, iloss=1)
        @test check_grad(opm(sqrt), (1.0, 2.0); verbose=true, iloss=1)
        @test check_grad(opm(abs), (1.0, -2.0); verbose=true, iloss=1)
        @test check_grad(opm(abs2), (1.0, -2.0); verbose=true, iloss=1)
        @test check_grad(opm(atan), (1.0, -2.0); verbose=true, iloss=1)
        @test check_grad(opm(atan), (1.0, -2.0, 1.5); verbose=true, iloss=1)
        @test check_grad(opm(exp), (1.0, 2.0); verbose=true, iloss=1)
        @test check_grad(opm(log), (1.0, 2.0); verbose=true, iloss=1)
        @test check_grad(opm(sin), (1.0, 2.0); verbose=true, iloss=1)
        @test check_grad(opm(tanh), (1.0, 2.0); verbose=true, iloss=1)
        @test check_grad(opm(cos), (1.0, 2.0); verbose=true, iloss=1)
        @test check_grad(opm(convert), (Fixed43(0.5), 2.0); verbose=true, iloss=1)
        @test check_grad(opm(/), (1.0, 2.0, 2.0); verbose=true, iloss=1)
        @test_broken check_grad(opm(÷), (1.0, 2.0, 2.0); verbose=true, iloss=1)
    end
    @test check_grad(-, (1.0,); verbose=true, iloss=1)
    @test check_grad(INC, (1.0,); verbose=true, iloss=1)
    @test check_grad(DEC, (1.0,); verbose=true, iloss=1)
    @test check_grad(ROT, (1.0, 2.0, 2.0); verbose=true, iloss=1)
    @test check_grad(ROT, (1.0, 2.0, 2.0); verbose=true, iloss=2)
    @test check_grad(IROT, (1.0, 2.0, 2.0); verbose=true, iloss=1)
    @test check_grad(IROT, (1.0, 2.0, 2.0); verbose=true, iloss=2)
end

@testset "sincos" begin
    @i function f(s, c, x)
        (s, c) += sincos(x)
    end
    @test check_grad(f, (1.0, 2.0, 2.0); verbose=true, iloss=1)
    @test check_grad(f, (1.0, 2.0, 2.0); verbose=true, iloss=2)
end

@testset "AD over pop" begin
    @i function mean(out!, x)
        anc ← zero(out!)
        for i=1:length(x)
            anc += x[i]
        end
        out! += anc / length(x)
        PUSH!(anc)
    end

    @test check_grad(mean, (0.0, [1,2,3.0, 4.0]); iloss=1)
end

@testset "AD over pipe" begin
    @i function mean(out!, anc, x)
        for i=1:length(x)
            ⊕(identity)(anc, x[i])
            SWAP(anc, x[i])
        end
        out! += anc / length(x)
    end

    @test check_grad(mean, (0.0, 0.0, [1,2,3.0, 4.0]); iloss=1, verbose=true)
end

@testset "push, load data" begin
    stack = []
    val = PUSH!(stack, [1,2,3])[2]
    @test val == Int[]
    val = PUSH!(stack, 3.0)[2]
    @test val == 0.0
    PUSH!(stack, 3.0)
    @test_throws InvertibilityError POP!(stack, GVar(3.0))
    PUSH!(stack, 3.0)
    @test POP!(stack, GVar(0.0))[2] == GVar(3.0)
    x = [1.0, 2.0, 3.0]
    PUSH!(stack, x)
    @test POP!(stack, empty(x))[2] == GVar.([1,2,3.0])
    x = [1.0, 2.0, 3.0]
    PUSH!(stack, x)
    @test POP!(stack, empty(x))[2] == [1,2,3.0]
end

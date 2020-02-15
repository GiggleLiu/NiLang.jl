using Test, NiLang, NiLang.AD
import NiLang: ⊕, ⊖

const add = ⊕(identity)

@testset "NGrad" begin
    @test exp''' isa NGrad{3,typeof(exp)}
    @test NGrad{3}(exp) isa NGrad{3,typeof(exp)}
    @test (~NGrad{2})(exp''') isa NGrad{1,typeof(exp)}
    @test (~NGrad{3})(exp''') === exp
end

@testset "instr" begin
    x, y = 3.0, 4.0
    lx = Loss(x)
    @instr (add)'(lx, y)
    @test grad(lx) == 1.0
    @test grad(y) == 1.0
    @test check_inv((add)', (Loss(3.0), 4.0); verbose=true, atol=1e-5)
    x, y = 3.0, 4.0
    lx = Loss(x)
    @test check_grad(add, (lx, y))

    x, y = 3.0, 4.0
    (add)'(Loss(x), NoGrad(y))
    @test grad(y) === 0.0

    @test check_inv(⊕(*), (0.4, 0.4, 0.5))
    @test ⊖(*)(GVar(0.0, 1.0), GVar(0.4), GVar(0.6)) == (GVar(-0.24, 1.0), GVar(0.4, 0.6), GVar(0.6, 0.4))
    @test check_grad(⊕(*), (Loss(0.4), 0.4, 0.5))
    @test check_grad(⊖(*), (Loss(0.4), 0.4, 0.5))
end

@testset "i" begin
    @i function test1(a, b, out)
        a ⊕ b
        out += a * b
    end

    @i function tt(a, b)
        out ← 0.0
        test1(a, b, out)
        (~test1)(a, b, out)
        a ⊕ b
    end

    # compute (a+b)*b -> out
    x = 3.0
    y = 4.0
    out = 0.0
    @test check_grad(test1, (x, y, Loss(out)))
    @test check_grad(tt, (Loss(x), y))
end


@testset "broadcast" begin
    # compute (a+b)*b -> out
    @i function test1(a, b)
        a .⊕ b
    end
    @i function test2(a, b, out, loss)
        a .⊕ b
        out .+= (a .* b)
        loss ⊕ out[1]
    end

    x = [3, 1.0]
    y = [4, 2.0]
    out = [0.0, 1.0]
    loss = 0.0
    # gradients
    @test check_grad(test2, (x, y, out, Loss(loss)))
end

@testset "broadcast 2" begin
    # compute (a+b)*b -> out
    @i function test1(a, b)
        a ⊕ b
    end
    @i function test2(a, b, out)
        a ⊕ b
        out += (a * b)
    end

    # gradients
    a = 1.0
    b = 1.3
    c = 1.9
    @test check_grad(test2, (a,b,Loss(c)))

    x = GVar([3, 1.0])
    y = GVar([4, 2.0])
    lout = GVar.([0.0, 1.0], [0.0, 2.0])
    @instr (~test2).(x, y, lout)
    @test grad.(lout) == [0,2.0]
    @test grad.(x) == [0, 4.0]
    @test grad.(y) == [0, 6.0]
end

@testset "function call function" begin
    # compute (a+b)*b -> out
    @i function test1(a, b)
        a ⊕ b
    end

    @i function test2(a, b, out)
        test1(a, out)
        (~test1)(a, out)
        out += (a * b)
    end

    a = 1.0
    b = 1.3
    c = 1.9
    @test check_grad(test2, (a,b,Loss(c)))
end

@testset "neg sign" begin
    @i function test(out, x, y)
        out += x * (-y)
    end
    @test check_grad(test, (Loss(0.1), 2.0, -2.5); verbose=true)
end

@testset "i" begin
    @i function test1(a::T, b, out) where T<:Number
        add(a, b)
        out += a * b
    end

    # compute (a+b)*b -> out
    @test isreversible(test1')
    @test isreversible(~test1')
    @test (~test1)' != ~(test1') # this is not true
end

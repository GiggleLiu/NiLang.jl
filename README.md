# NiLang

Note: this project is still in progress, with a lot of unstable features.
Please read the tests in `NiLangCore.jl` and `NiLang.jl` to figure out the tested features.

NiLang.jl (逆lang), is a reversible domain sepecific language (DSL) in Julia.
It features

* An instruction level (i.e. only backward rules of `+`, `-`, `*` and `/` are required) automatic differentiation engine,
* A reversible language with abstraction and arrays,
* Arbituary high order gradint (in progress).

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://GiggleLiu.github.io/NiLang.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://GiggleLiu.github.io/NiLang.jl/dev)
[![Build Status](https://travis-ci.com/GiggleLiu/NiLang.jl.svg?branch=master)](https://travis-ci.com/GiggleLiu/NiLang.jl)
[![Codecov](https://codecov.io/gh/GiggleLiu/NiLang.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/GiggleLiu/NiLang.jl)

## To Start
```
pkg> dev git@github.com:GiggleLiu/NiLangCore.jl.git
pkg> dev git@github.com:GiggleLiu/NiLang.jl.git
```

## Examples
1. a unitary matrix multiplication operation, parametrized by at most `N*(N+1)/2` parameters (θ).
```julia
@i function umm!(x, θ, Nin::Int, Nout::Int)
    @anc k = 0
    for j=1:Nout
        for i=Nin-1:-1:j
            k += 1
            ROT(x[i], x[i+1], θ[k])
        end
    end

    # uncompute k
    for j=1:Nout
        for i=Nin-1:-1:j
            k -= 1
        end
    end
end
```

Notes:
* `+=` and `-=` are inplace `+` and `-` operations.
In fact, all instructions/function in `NiLang` are (effectively) inplace.
* `@anc x = val` declares an ancilla register with initial value `val`.
At the end of computation, this ancilla register must be reset to `0` and return to system,
otherwise raises `InvertibilityError`.
* control flow `for start:step:stop ... end` looks quite similar to traditional computation,
except it errors if `start`, `step` or `stop` changes during computation.
* `if` and `while` statement is a bit different, they use a tuple of (precondition, postcondition) as input. precondition is used in forward execution, postcondition is used in backward execution.
These two conditions should match, otherwise errors.

2. The autodiff engine
```julia
# compute `((a.+b).*b)[1] -> out`
@i function test1(a, b)
    a .+= b
end
@i function test2(a, b, out, loss)
    a .+= b
    out .+= (a .* b)
    loss += out[1]
end

x = [3, 1.0]
y = [4, 2.0]
out = [0.0, 1.0]
loss = 0.0
# gradients
@instr test2'(x, y, out, loss)
ga = grad(x)

# to reclaim gradient memory in reversible computer, one should call
@instr (~test2')(x, y, out, loss)
```

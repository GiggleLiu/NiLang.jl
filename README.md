# NiLang

Warning: 

* This project is still in progress, with a lot of unstable features.
Please read the tests in `NiLangCore.jl` and `NiLang.jl` to figure out the tested features.
* Requires Julia version >= 1.3.


NiLang.jl (逆lang), is a reversible domain sepecific language (DSL) in Julia.
It features:

* an instruction level (i.e. only backward rules of `+`, `-`, `*` and `/` are required) automatic differentiation engine,
* a reversible language with abstraction and arrays,
* arbituary high order gradint (in progress).

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://GiggleLiu.github.io/NiLang.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://GiggleLiu.github.io/NiLang.jl/dev)
[![Build Status](https://travis-ci.com/GiggleLiu/NiLang.jl.svg?branch=master)](https://travis-ci.com/GiggleLiu/NiLang.jl)
[![Codecov](https://codecov.io/gh/GiggleLiu/NiLang.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/GiggleLiu/NiLang.jl)

> The strangeness of reversible computing is mainly due to
> our lack of experience with it.—Henry Baker, 1992

## To Start
```
pkg> dev git@github.com:GiggleLiu/NiLangCore.jl.git
pkg> dev git@github.com:GiggleLiu/NiLang.jl.git
```

## Examples
1. Compute exp function from Taylor expansion

```julia
@i function iexp(out!, x::T; atol::Float64=1e-14) where T
    @anc anc1 = zero(T)
    @anc anc2 = zero(T)
    @anc anc3 = zero(T)
    @anc iplus = 0
    @anc expout = zero(T)

    out! += identity(1.0)
    @routine r1 begin
        anc1 += identity(1.0)
        while (value(anc1) > atol, iplus != 0)
            iplus += identity(1)
            anc2 += anc1 * x
            anc3 += anc2 / iplus
            expout += identity(anc3)
            # speudo inverse
            anc1 -= anc2 / x
            anc2 -= anc3 * iplus
            SWAP(anc1, anc3)
        end
    end

    out! += identity(expout)

    ~@routine r1
end
```

To understand the grammar, see the [README](https://github.com/GiggleLiu/NiLangCore.jl) of NiLangCore.

2. The autodiff engine

```julia
julia> y!, x = 0.0, 1.6
(0.0, 1.6)

# first order gradients
julia> @instr iexp'(Loss(y!), x)

julia> grad(x)
4.9530324244260555

julia> y!, x = 0.0, 1.6
(0.0, 1.6)

# second order gradient by differentiate first order gradients
julia> simple_hessian(iexp, (Loss(y!), x))
2×2 Array{Float64,2}:
0.0 0.0
0.0 4.95303

julia> y!, x = 0.0, 1.6
(0.0, 1.6)

# second order gradient by taylor propagation (experimental)
julia> @instr iexp''(Loss(y!), x)

julia> collect_hessian()
2×2 Array{Float64,2}:
0.0 0.0
0.0 4.95303
```

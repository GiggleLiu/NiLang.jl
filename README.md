# NiLang

NiLang.jl (逆lang), is a reversible domain-specific language (DSL) that allow a program to go back to the past.

<img src="docs/src/asset/logo2.jpg" width=500px/>

* Requires Julia version >= 1.3.
* If test breaks, try using the master branch of `NiLangCore`.


NiLang features:

* any program written in NiLang is differentiable,
* a reversible language with abstraction and arrays,
* complex values

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://GiggleLiu.github.io/NiLang.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://GiggleLiu.github.io/NiLang.jl/dev)
[![Build Status](https://travis-ci.com/GiggleLiu/NiLang.jl.svg?branch=master)](https://travis-ci.com/GiggleLiu/NiLang.jl)
[![Codecov](https://codecov.io/gh/GiggleLiu/NiLang.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/GiggleLiu/NiLang.jl)

> The strangeness of reversible computing is mainly due to
> our lack of experience with it.—Henry Baker, 1992

Please check [why reversible computing is the future of computing](https://giggleliu.github.io/NiLang.jl/dev/why/).

## To Start
```
pkg> add NiLangCore
pkg> add NiLang
```

## Examples
1. Compute exp function from Taylor expansion

```julia
using NiLang, NiLang.AD

@i function iexp(out!, x::T; atol::Real=1e-14) where T
    anc1 ← zero(T)
    anc2 ← zero(T)
    anc3 ← zero(T)
    iplus ← 0
    expout ← zero(T)

    out! += identity(1)
    @routine begin
        anc1 += identity(1)
        while (value(anc1) > atol, iplus != 0)
            INC(iplus)
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

    ~@routine
end
```

To understand the grammar, see the [README](https://github.com/GiggleLiu/NiLangCore.jl) of NiLangCore.

2. The autodiff engine

```julia
julia> y!, x = 0.0, 1.6
(0.0, 1.6)

# first order gradients
julia> @instr iexp'(Val(1), y!, x)

julia> grad(x)
4.9530324244260555

julia> y!, x = 0.0, 1.6
(0.0, 1.6)

# second order gradient by differentiate first order gradients
julia> using ForwardDiff: Dual

julia> _, hxy, hxx = iexp'(Val(1), 
        Dual(y!, zero(y!)), Dual(x, one(x)));

julia> grad(hxx).partials[1]
4.953032423978584
```

See [more examples](examples/)

## Cite our [paper](https://arxiv.org/abs/2003.04617)!

```bibtex
@misc{Liu2020,
    title={Differentiate Everything with a Reversible Programming Language},
    author={Jin-Guo Liu and Taine Zhao},
    year={2020},
    eprint={2003.04617},
    archivePrefix={arXiv},
    primaryClass={cs.PL}
}
```

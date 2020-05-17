# NiLang

NiLang.jl (逆lang), is a reversible domain-specific language (DSL) that allow a program to go back to the past.

<img src="docs/src/asset/logo3.png" width=300px/>

* Requires Julia version >= 1.3.
* If test breaks, try using the master branch of `NiLangCore`.
* **The `'` notation has been removed recently!**


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
1. Compute sparse matrix multiplication

```julia
using NiLang
using SparseArrays: SparseMatrixCSC, AbstractSparseMatrix, nonzeros, rowvals, getcolptr

@i function mul!(C::StridedVecOrMat, A::AbstractSparseMatrix, B::StridedVector{T}, α::Number, β::Number) where T
    @safe size(A, 2) == size(B, 1) || throw(DimensionMismatch())
    @safe size(A, 1) == size(C, 1) || throw(DimensionMismatch())
    @safe size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    nzv ← nonzeros(A)
    rv ← rowvals(A)
    if (β != 1, ~)
        @safe error("only β = 1 is supported, got β = $(β).")
    end
    # Here, we close the reversibility check inside the loop to increase performance
    @invcheckoff for k = 1:size(C, 2)
        @inbounds for col = 1:size(A, 2)
            αxj ← zero(T)
            αxj += B[col,k] * α
            for j = getcolptr(A)[col]:(getcolptr(A)[col + 1] - 1)
                C[rv[j], k] += nzv[j]*αxj
            end
            αxj -= B[col,k] * α
        end
    end
end
```

To back propagate the gradient
```
julia> using SparseArrays: sprand

julia> import SparseArrays

julia> using BenchmarkTools

julia> n = 1000;

julia> sp1 = sprand(ComplexF64, n, n,0.1);

julia> v = randn(ComplexF64, n);

julia> out = zero(v);

julia> @benchmark SparseArrays.mul!($(copy(out)), $sp1, $v, 0.5+0im, 1)
BenchmarkTools.Trial: 
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     226.005 μs (0.00% GC)
  median time:      235.590 μs (0.00% GC)
  mean time:        244.868 μs (0.00% GC)
  maximum time:     2.750 ms (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     1

julia> @benchmark mul!($(copy(out)), $(sp1), $v, 0.5+0im, 1)
BenchmarkTools.Trial: 
  memory estimate:  64 bytes
  allocs estimate:  1
  --------------
  minimum time:     194.011 μs (0.00% GC)
  median time:      207.218 μs (0.00% GC)
  mean time:        257.364 μs (0.00% GC)
  maximum time:     2.324 ms (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     1

julia> using NiLang.AD

julia> @benchmark (~mul!)($(GVar(copy(out))), $(GVar(sp1)), $(GVar(v)), $(GVar(0.5)), 1)
BenchmarkTools.Trial: 
  memory estimate:  64 bytes
  allocs estimate:  1
  --------------
  minimum time:     719.223 μs (0.00% GC)
  median time:      767.744 μs (0.00% GC)
  mean time:        790.198 μs (0.00% GC)
  maximum time:     7.231 ms (0.00% GC)
  --------------
  samples:          6291
  evals/sample:     1
```

To understand the grammar, see the [README](https://github.com/GiggleLiu/NiLangCore.jl) of NiLangCore.

2. The autodiff engine

```julia
julia> y!, x = 0.0, 1.6
(0.0, 1.6)

# first order gradients
julia> @instr Grad(iexp)(Val(1), y!, x)

julia> grad(x)
4.9530324244260555

julia> y!, x = 0.0, 1.6
(0.0, 1.6)

# second order gradient by differentiate first order gradients
julia> using ForwardDiff: Dual

julia> _, hxy, hxx = Grad(iexp)(Val(1), 
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

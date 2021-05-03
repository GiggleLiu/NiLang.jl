# How to extend

## Extend `+=`, `-=` and `⊻=` for irreversible one-out functions

It directly works
```julia
julia> using SpecialFunctions, NiLang

julia> x, y = 2.1, 1.0
(2.1, 1.0)

julia> @instr y += besselj0(x)
2.1

julia> x, y
(2.1, 1.7492472503018073)

julia> @instr ~(y += besselj0(x))
2.1

julia> x, y
(2.1, 1.0)
```

Here the statement
```julia
@instr y += besselj0(x)
```

is mapped to
```julia
@instr y += besselj0(x)
```

However, doing this does not give you correct gradients.
For `y += scalar_out_function(x)`, one can bind the backward rules like

```julia
julia> using ChainRules, NiLang.AD

julia> besselj0_back(x) = ChainRules.rrule(besselj0, x)[2](1.0)[2]
besselj0_back (generic function with 1 method)

julia> primitive_grad(::typeof(besselj0), x::Real) = besselj0_back(x)
primitive_grad (generic function with 1 method)

julia> xg, yg = GVar(x), GVar(y, 1.0)
(GVar(2.1, 0.0), GVar(1.0, 1.0))

julia> @instr yg -= besselj0(xg)
GVar(2.1, -0.5682921357570385)

julia> xg, yg
(GVar(2.1, -0.5682921357570385), GVar(0.8333930196680097, 1.0))

julia> @instr yg += besselj0(xg)
GVar(2.1, 0.0)

julia> xg, yg
(GVar(2.1, 0.0), GVar(1.0, 1.0))

julia> NiLang.AD.check_grad(PlusEq(besselj0), (1.0, 2.1); iloss=1)
true

julia> using BenchmarkTools

julia> @benchmark PlusEq(besselj0)($yg, $xg)
BenchmarkTools.Trial: 
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     451.523 ns (0.00% GC)
  median time:      459.431 ns (0.00% GC)
  mean time:        477.419 ns (0.00% GC)
  maximum time:     857.036 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     197
```

Good!

## Reversible multi-in multi-out functions

It is easy to do, define two normal Julia functions reversible to each other,
using the macro `@dual` to tell the compiler they are reversible to each other.

For example, a pair of dual functions `ROT` (2D rotation) and `IROT` (inverse rotation) that already defined in NiLang.

```julia
"""
    ROT(a!, b!, θ) -> a!', b!', θ
"""
@inline function ROT(i::Real, j::Real, θ::Real)
    a, b = rot(i, j, θ)
    a, b, θ
end

"""
    IROT(a!, b!, θ) -> ROT(a!, b!, -θ)
"""
@inline function IROT(i::Real, j::Real, θ::Real)
    i, j, _ = ROT(i, j, -θ)
    i, j, θ
end
@dual ROT IROT
```

One can easily check the reversibility by typing
```julia
julia> check_inv(ROT, (1.0, 2.0, 3.0))
true
```

For self-reversible functions, one can declare the reversibility for it like this
```julia
"""
    SWAP(a!, b!) -> b!, a!
"""
@inline function SWAP(a!::Real, b!::Real)
    b!, a!
end
@selfdual SWAP
```

To bind gradients for this multi-in, multi-out function.
The general approach is *Binding the backward rule on its inverse*!

```julia
@i @inline function IROT(a!::GVar, b!::GVar, θ::GVar)
    IROT(a!.x, b!.x, θ.x)
    NEG(θ.x)
    θ.x -= π/2
    ROT(a!.g, b!.g, θ.x)
    θ.g += a!.x * a!.g
    θ.g += b!.x * b!.g
    θ.x += π/2
    NEG(θ.x)
    ROT(a!.g, b!.g, π/2)
end

@i @inline function IROT(a!::GVar, b!::GVar, θ::Real)
    IROT(a!.x, b!.x, θ)
    NEG(θ)
    θ -= π/2
    ROT(a!.g, b!.g, θ)
    θ += π/2
    NEG(θ)
    ROT(a!.g, b!.g, π/2)
end

@nograd IROT(a!::Real, b!::Real, θ::GVar)
```

When this inverse function is called, the backward rules are automatically applied.

Good! This method can also be extended to linear algebra functions, however, the memory allocation overhead is high because one need to wrap each element with `GVar`.

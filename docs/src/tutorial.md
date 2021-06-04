# My first NiLang program

## Basic Statements

| Statement                 | Meaning                                                      |
| :------------------------ | :----------------------------------------------------------- |
| x ← val                   | allocate a new variable `x`, with an initial value `val` (a constant). |
| x → val                   | deallocate variable `x` with content `val`.                  |
| x += f(y)                 | a reversible instruction.                                    |
| x .+= f.(y)                | instruction call with broadcasting.                          |
| f(y)                      | a reversible function.                                       |
| f.(y)                     | function call with broadcasting.                             |
| if (pre, post) ... end    | if statement.                                                |
| @from post while pre ... end | while statement.                                             |
| for x=1:3 ... end         | for statement.                                               |
| begin ... end             | block statement.                                             |
| @safe ...                 | insert an irreversible statement.                            |
| ~(...)                    | inverse a statement.                                         |
| @routine ...              | record a routine in the **routine stack**.                   |
| ~@routine                 | place the inverse of the routine on **routine stack** top.   |

The condition expression in **if** and **while** statements are a bit hard to digest, please refer our paper [arXiv:2003.04617](https://arxiv.org/abs/2003.04617).

## A reversible program

Our first program is to compute a loss function defined as

```math
\mathcal{L} = {\vec z}^T(a\vec{x} + \vec{y}),
```

where $\vec x$, $\vec y$ and $\vec{z}$ are column vectors, $a$ is a scalar.

```julia
@i function r_axpy!(a::T, x::AbstractVector{T}, y!::AbstractVector{T}) where T
    @safe @assert length(x) == length(y!)
    for i=1:length(x)
        y![i] += a * x[i]
    end
end

@i function r_loss(out!, a, x, y!, z)
    r_axpy!(a, x, y!)
    for i=1:length(z)
    	out! += z[i] * y![i]
    end
end
```

First let's explain the codes a bit. `@i` macro is the core of NiLang; all reversible
functions in NiLang has this decorator. By adding this macro you're promising that
`r_axpy!` is a reversible function. In other words, the composition of `r_axpy!` and
its inverse `~r_axpy!` becomes an identity map, i.e., `(~r_axpy!)(r_axpy!(args...)...) ≈ args`
for all valid input `args`.

Some functions and variables in NiLang will ends with `!`. This is a Julia convention saying
that `r_axpy!` is an in-place function which modifies the input, and that input `y!` and `out!`
will be modified.

Perhaps surprisingly, you can't write an explicit `return` in `@i`. This is because NiLang's
reversible programming requires a complete "compute–copy–uncompute" paradigm. NiLang is sometimes
smart enough to infer the copy and uncompute stage so you won't need to manually write it. A
complete version of `r_axpy!` is:

```julia
@i function r_axpy!(a::T, x::AbstractVector{T}, y!::AbstractVector{T}) where T
    @safe @assert length(x) == length(y!)
    # compute
    @routine begin
        for i=1:length(x)
            y![i] += a * x[i]
        end
    end

    # no copy operation here

    # uncompute
    ~@routine

    # `@i` forces returning all input variables as outputs, i.e.,
    # return a, x, y!
    # and you can't override this
end
```

Functions do not have return statements, they return all input arguments instead.
Hence `r_loss` defines a 5 variable to 5 variable bijection.
Let's check the reversibility
```julia
julia> out, a, x, y, z = 0.0, 2.0, randn(3), randn(3), randn(3)
(0.0, 2.0, [0.9265845776642722, 0.8532458027149912, 0.6201064385679095],
 [1.1142808415540468, 0.5506163710455121, -1.9873779917908814],
 [1.1603953198942412, 0.5562855137395296, 1.9650050430758796])

julia> out, a, x, y, z = r_loss(out, a, x, y, z)
(3.2308283403544342, 2.0, [0.9265845776642722, 0.8532458027149912, 0.6201064385679095],
 [2.967449996882591, 2.2571079764754947, -0.7471651146550624],
 [1.1603953198942412, 0.5562855137395296, 1.9650050430758796])
```

We find the contents in `out` and `y` are changed after calling the loss function.
If we call the inverse loss function `~r_loss`, then values are restored:

```julia
julia> out, a, x, y, z = (~r_loss)(out, a, x, y, z)
(0.0, 2.0, [0.9265845776642722, 0.8532458027149912, 0.6201064385679095],
 [1.1142808415540466, 0.5506163710455123, -1.9873779917908814],
 [1.1603953198942412, 0.5562855137395296, 1.9650050430758796])
```

Here, instead of assigning variables one by one,
one can also use the macro `@instr`
```julia
@instr r_loss(out, a, x, y, z)
```
`@instr` macro is for executing a reversible statement.

Let's go back a bit and restate the "compute-copy-uncompute" paradigm. It is sometimes
tedious and cumbersome to write an uncompute manually. NiLang introduces `@routine` macro
to record all operations, and automatically reverse these operations with `~@routine`.
For this reason, each compute stage `@routine` should always have a corresponding
uncompute stage `~@routine`:

!!! tip
    You can intuitively take `@routine` as a special 0-argument function that only lives
    in the `@i` scope, and `~@routine` is its inverse.

```julia
julia> @i function r_axpy!(a::T, x::AbstractVector{T}, y!::AbstractVector{T}) where T
    @safe @assert length(x) == length(y!)
    # compute
    @routine begin
        for i=1:length(x)
            y![i] += a * x[i]
        end
    end
    # no corresponding uncompute stage
end
ERROR: LoadError: `@routine` and `~@routine` must appear in pairs, mising `~@routine`!
Stacktrace:
[...]
```

## My first reversible AD program

```julia
julia> using NiLang.AD: Grad

julia> x, y, z = randn(3), randn(3), randn(3)
([2.2683181471139906, -0.7374245775047469, 0.9568936661385092],
 [1.0275914704043452, 1.647972121962081, -0.8349079845797637],
 [1.4272076815911372, 0.5317755971532034, 0.4412421572457776])

julia> Grad(r_loss)(0.0, 0.5, x, y, z; iloss=1)
(GVar(0.0, 1.0), GVar(0.5, 3.2674385142974036),
 GVar{Float64,Float64}[GVar(2.2683181471139906, 0.7136038407955686), GVar(-0.7374245775047469, 0.2658877985766017), GVar(0.9568936661385092, 0.2206210786228888)],
 GVar{Float64,Float64}[GVar(2.1617505439613405, 1.4272076815911372), GVar(1.2792598332097076, 0.5317755971532034), GVar(-0.35646115151050906, 0.4412421572457776)],
 GVar{Float64,Float64}[GVar(1.4272076815911372, 3.295909617518336), GVar(0.5317755971532034, 0.9105475444573341), GVar(0.4412421572457776, 0.12198568155874556)])

julia> gout, ga, gx, gy, gz = Grad(r_loss)(0.0, 0.5, x, y, z; iloss=1)
(GVar(0.0, 1.0), GVar(0.5, 3.2674385142974036),
 GVar{Float64,Float64}[GVar(2.2683181471139906, 0.7136038407955686), GVar(-0.7374245775047469, 0.2658877985766017), GVar(0.9568936661385092, 0.2206210786228888)],
 GVar{Float64,Float64}[GVar(3.295909617518336, 1.4272076815911372), GVar(0.9105475444573341, 0.5317755971532034), GVar(0.12198568155874556, 0.4412421572457776)],
 GVar{Float64,Float64}[GVar(1.4272076815911372, 4.4300686910753315), GVar(0.5317755971532034, 0.5418352557049606), GVar(0.4412421572457776, 0.6004325146280002)])
```

The results are a bit messy, since NiLang wraps each element with a gradient field automatically. We can take the gradient field using the `grad` function like

```julia
julia> grad(gout)
1.0

julia> grad(ga)
3.2674385142974036

julia> grad(gx)
3-element Array{Float64,1}:
 0.7136038407955686
 0.2658877985766017
 0.2206210786228888

julia> grad(gy)
3-element Array{Float64,1}:
 1.4272076815911372
 0.5317755971532034
 0.4412421572457776

julia> grad(gz)
3-element Array{Float64,1}:
 4.4300686910753315
 0.5418352557049606
 0.6004325146280002
```

## Writing irreversible function in a reversible way

Not all functions are reversible: operations that erases the compute history is not reversible
operations. For example, in general:

- `x *= 0` reset `x` to be zero and thus `*` (and `/`) are not a reversible operations;
- shared read/write operation `y += f(y)` clears the old status of `y` and is not reversible

But in practice, you can always rewrite them in a reversible way.

The first trick is to _use extra bits to record the intermediate results_:

```julia
@i function i_wsqeuclidean(out!::T, X::AbstractArray{T}, Y::AbstractArray{T}, W::AbstractArray{T}) where T
    @safe @assert size(W) == size(X) == size(Y)
    for i = 1:length(X)
        # compute stage
        @routine begin
            @zeros T d d2
            # All intermediate results need to be recorded in the compute
            # stage so that they can be successfully uncomputed.
            d += X[i] - Y[i]
            d2 += abs2(d)
        end

        # copy
        out! += d2 * W[i]

        # uncompute stage reverses the computation to its initial status,
        # it also restores intermediate results `d` and `d2` to zero value
        ~@routine
    end
end
```

Don't worry if you don't know the whole set of irreversible operations, when constructing function
with `@i`, a reversibility check will be called and throw errors when the function body is not
reversible. Hence you're always in a safe status.

```julia
julia> @i function irreversible_f(out!)
           out! += abs2(out!)
       end
ERROR: LoadError: InvertibilityError("1-th argument and 2-th argument shares the same memory out!, shared read and shared write are not allowed!")
```

Of course, reversibility check takes time, and the overhead and be quite significant in very tight loops,
take our previous `i_wsqeuclidean` as an example, `length(X)` reversible checks are applied here.

```julia
using Benchmark

X, Y, W = rand(5, 5), rand(5, 5), rand(5, 5)
@btime i_wsqeuclidean(0.0, X, Y, W)[1]
# 1.122 μs (2 allocations: 64 bytes)
```

The macro `@invcheckoff` can be used to disable the reversibility check for the entire block.
For example, by adding it to the for-loop block, all checks are disabled.

```julia
@i function i_wsqeuclidean(out!::T, X::AbstractArray{T}, Y::AbstractArray{T}, W::AbstractArray{T}) where T
    @safe @assert size(W) == size(X) == size(Y)
    @invcheckoff for i = 1:length(X)
        @routine begin
            @zeros T d d2
            # All intermediate results need to be recorded in the compute
            # stage so that they can be successfully uncomputed.
            d += X[i] - Y[i]
            d2 += abs2(d)
        end
        out! += d2 * W[i]

        # uncompute stage reverses the computation to its initial status,
        # it also restores intermediate results `d` and `d2` to zero value
        ~@routine
    end
end
```

it's significantly faster now:

```julia
@btime i_wsqeuclidean(0.0, X, Y, W)[1]
# 67.269 ns (2 allocations: 64 bytes)
```

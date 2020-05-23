# Tutorial

## Basic Statements

| Statement                 | Meaning                                                      |
| ------------------------- | ------------------------------------------------------------ |
| x ← val                   | allocate a new variable `x`, with an initial value `val` (a constant). |
| x → val                   | deallocate variable `x` with content `val`.                  |
| x += f(y)                 | a reversible instruction.                                    |
| x .+= f(y)                | instruction call with broadcasting.                          |
| f(y)                      | a reversible function.                                       |
| f.(y)                     | function call with broadcasting.                             |
| if (pre, post) ... end    | if statement.                                                |
| while (pre, post) ... end | while statement.                                             |
| for x=1:3 ... end         | for statement.                                               |
| begin ... end             | block statement.                                             |
| @safe ...                 | insert an irreversible statement.                            |
| ~(...)                    | inverse a statement.                                         |
| @routine ...              | record a routine in the **routine stack**.                   |
| ~@routine                 | place the inverse of the routine on **routine stack** top.   |

The condition in if and while statements are a bit hard to digest, please refer our paper [arXiv:2003.04617](https://arxiv.org/abs/2003.04617).

## My first NiLang program

To compute a loss
$$
\mathcal{L} = {\vec z}^T(a\vec{x} + \vec{y})
$$
Where $\vec x$, $\vec y$ and $\vec{z}$ are column vectors, $a$ is a scalar.

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

Functions do not have return statements, they return input arguments instead.
Hence `r_loss` defines a 5 variable to 5 variable bijection.
Let's check the reversibility
```julia
julia> out, a, x, y, z = 0.0, 2.0, randn(3), randn(3), randn(3)
(0.0, 2.0, [0.9265845776642722, 0.8532458027149912, 0.6201064385679095], [1.1142808415540468, 0.5506163710455121, -1.9873779917908814], [1.1603953198942412, 0.5562855137395296, 1.9650050430758796])

julia> out, a, x, y, z = r_loss(out, a, x, y, z)
(3.2308283403544342, 2.0, [0.9265845776642722, 0.8532458027149912, 0.6201064385679095], [2.967449996882591, 2.2571079764754947, -0.7471651146550624], [1.1603953198942412, 0.5562855137395296, 1.9650050430758796])
```

We find the contents in `out` and `y` are changed after calling the loss function.
Then we call the inverse loss function `~r_loss`.

```julia
julia> out, a, x, y, z = (~r_loss)(out, a, x, y, z)
(0.0, 2.0, [0.9265845776642722, 0.8532458027149912, 0.6201064385679095], [1.1142808415540466, 0.5506163710455123, -1.9873779917908814], [1.1603953198942412, 0.5562855137395296, 1.9650050430758796])
```

Values are restored.


## My first reversible AD program

```julia
julia> using NiLang.AD: Grad

julia> x, y, z = randn(3), randn(3), randn(3)
([2.2683181471139906, -0.7374245775047469, 0.9568936661385092], [1.0275914704043452, 1.647972121962081, -0.8349079845797637], [1.4272076815911372, 0.5317755971532034, 0.4412421572457776])

julia> Grad(r_loss)(0.0, 0.5, x, y, z; iloss=1)
(GVar(0.0, 1.0), GVar(0.5, 3.2674385142974036), GVar{Float64,Float64}[GVar(2.2683181471139906, 0.7136038407955686), GVar(-0.7374245775047469, 0.2658877985766017), GVar(0.9568936661385092, 0.2206210786228888)], GVar{Float64,Float64}[GVar(2.1617505439613405, 1.4272076815911372), GVar(1.2792598332097076, 0.5317755971532034), GVar(-0.35646115151050906, 0.4412421572457776)], GVar{Float64,Float64}[GVar(1.4272076815911372, 3.295909617518336), GVar(0.5317755971532034, 0.9105475444573341), GVar(0.4412421572457776, 0.12198568155874556)])

julia> gout, ga, gx, gy, gz = Grad(r_loss)(0.0, 0.5, x, y, z; iloss=1)
(GVar(0.0, 1.0), GVar(0.5, 3.2674385142974036), GVar{Float64,Float64}[GVar(2.2683181471139906, 0.7136038407955686), GVar(-0.7374245775047469, 0.2658877985766017), GVar(0.9568936661385092, 0.2206210786228888)], GVar{Float64,Float64}[GVar(3.295909617518336, 1.4272076815911372), GVar(0.9105475444573341, 0.5317755971532034), GVar(0.12198568155874556, 0.4412421572457776)], GVar{Float64,Float64}[GVar(1.4272076815911372, 4.4300686910753315), GVar(0.5317755971532034, 0.5418352557049606), GVar(0.4412421572457776, 0.6004325146280002)])
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

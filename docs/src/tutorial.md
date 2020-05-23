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


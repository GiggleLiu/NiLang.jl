# variable manipulation
export @zeros, @ones

"""
Create zeros of specific type.

```julia
julia> @i function f(x)
           @zeros Float64 a b c
           # do something
       end
```
"""
macro zeros(T, args...)
    esc(Expr(:block, map(x->:($x ← zero($T)), args)...))
end

macro ones(T, args...)
    esc(Expr(:block, map(x->:($x ← one($T)), args)...))
end

function NiLangCore.chfield(a::AbstractArray, ::typeof(vec), val)
    reshape(val, size(a)...)
end

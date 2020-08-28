## Why reversibility check fails even though the program is reversible?
Due to the fact that floating pointing numbers are not exactly reversible, sometimes the invertibility check might fail due to the rounding error.

To fix this issue, you may want to make the check less restrictive
```julia
NiLangCore.GLOBAL_ATOL[] = 1e-6  # default is 1e-8
```

Or just turn off the check in the program (only if you are sure the program is correct)
```julia
@routine @invcheckoff begin
    ...
end
```
Turning off the check will make your program faster too!

## What makes the gradient check fails?
##### Finite difference error due to numeric instability
The `NiLang.AD.check_grad` function sometimes fail due to either the rounding error or the finite difference error, you may want to check the gradient manually with the `NiLang.AD.ng` function (numeric gradient).
```julia
julia> NiLang.AD.ng(jin, copy.((out,b,ma,jinzhi,spread,bili)), 6; iloss=1, δ=1e-4)
-5449.643843214744

julia> NiLang.AD.ng(jin, copy.((out,b,ma,jinzhi,spread,bili)), 5; iloss=1, δ=1e-4)
4503-element Array{Float64,1}:
 -0.0023380584934784565
 -0.0021096593627589755
 -0.0019811886886600405
  ⋮
 -0.009526640951662557
 -0.006004695478623034
  0.0
```

and 
```julia
julia> NiLang.AD.gradient(Val(1), jin, copy.((out,b,ma,jinzhi,spread,bili)))[end]
-5449.643116967733

julia> NiLang.AD.gradient(Val(1), jin, copy.((out,b,ma,jinzhi,spread,bili)))[end-1]
4503-element Array{Float64,1}:
 -0.0005285958114468947
 -0.00030225263725219137
 -0.00017545437275561654
  ⋮
 -0.010422627668532736
 -0.0069140339974312695
  0.0
```

Here, we can see the `jin` function is numerically sensitive to perturbations, which makes the numeric gradient incorrect.
The above code is from https://github.com/HanLi123/NiLang/issues/3

##### Allocating a non-constant ancilla
Another possibility is, a non-constant ancilla is allocated.

```julia
julia> @i function f1(z, y)
           x ← y   # wrong!
           z += x
           x → y
       end

julia> NiLang.AD.gradient(Val(1), f1, (0.0, 1.0))
(1.0, 0.0)

julia> @i function f2(z, y)
           x ← zero(y)
           x += y
           z += x
           x -= y
           x → zero(y)
       end

julia> NiLang.AD.gradient(Val(1), f2, (0.0, 1.0))
(1.0, 1.0)
```
`f1` will give incorrect gradient because when ancilla `x` is deallocated, its gradient field will also be discarded.

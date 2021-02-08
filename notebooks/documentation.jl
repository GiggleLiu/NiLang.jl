### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ d941d6c2-55bf-11eb-0002-35c7474e4050
using NiLang, Test

# ╔═╡ 8c2c4fa6-172f-4dde-a279-5d0aecfdbe46
module M
using NiLang

# define two functions
function new_forward(x)
	if x > 0
		return x * 2
	elseif x < 0
		return x / 2
	end
end

function new_backward(x)
	if x > 0
		return x / 2
	elseif x < 0
		return x * 2
	end
end

# declare them as reversible to each other
@dual new_forward new_backward

using NiLang.AD: GVar
function new_backward(x::GVar)
	if x.x > 0
		GVar(new_backward(x.x), x.g * 2)
	elseif x.x < 0
		GVar(new_backward(x.x), x.g / 2)
	end
end

function new_forward(x::GVar)
	if x.x > 0
		GVar(new_forward(x.x), x.g / 2)
	elseif x.x < 0
		GVar(new_forward(x.x), x.g * 2)
	end
end
end

# ╔═╡ 3199a048-7b39-40f8-8183-6a54cccd91b6
using BenchmarkTools

# ╔═╡ 2061b434-0ad1-46eb-a0c7-1a5f432bfa62
begin
	twocol(left, right; llabel="forward", rlabel="backward") = HTML("
<table style=\"border:0px\"  class=\"normal-table\" width=80%>
	<tr>
	<td align='center' style='font-family:verdana; background-color: white;'>$llabel</td>
	<td align='center' style='font-family:verdana; background-color: white;'>$rlabel</td>
	</tr>
<tr style=\"background-color: white;\">
<td>
$(html(left))
</td>
<td>
$(html(right))
</td>
</tr>
</table>
")
	example(str) = HTML("""<h6 class="example">$str</h6>""")
	title1(str) = HTML("""<div class="root"><h2 id=$(replace(str, ' '=>'_'))>$str</h2></div>""")
	title2(str) = HTML("""<h4 id=$(replace(str, ' '=>'_'))>$str</h4>""")
	titleref(str) = HTML("""<a href="#$(replace(str, ' '=>'_'))">$str</a>""")
end;

# ╔═╡ a4e76427-f051-4b29-915a-fdfce3a299bb
html"""
<style>

body {
counter-reset: section subsection example}

h2::before {
counter-reset: subsection;
  counter-increment: section;
  content: "Section " counter(section) ": ";
}
h4::before {
  counter-increment: subsection;
  content: "⋄ ";
}

h6.example::before {
  counter-increment: example;
  content: "Example "counter(example) ": ";
}
</style>
"""

# ╔═╡ c2c7b4d4-f8c9-4ebf-8da2-0103f03136e7
md"# NiLang's Documentation (v0.8.0)

NiLang is a embeded domain specific language (eDSL) in Julia, so one need to install [Julia](https://julialang.org/) first. Before reading this documentation, you need to know basic Julia grammar, and how to install and use packages.
Also, we preassume a user already knows what they can get from NiLang, if not, please read the [README](https://github.com/GiggleLiu/NiLang.jl). In this tutorial, we focus on

* NiLang grammar and design patterns.
* Automatic differentiation based on reversible programming and time-space tradeoff
"

# ╔═╡ 605872cf-f3fd-462e-a2b1-7d1c5ae45efd
title1("Get started")

# ╔═╡ fb3dee44-5fa9-4773-8b7f-a83c44358545
md"
After installing NiLang in a Julia REPL by typing `]add NiLang`, one can use NiLang and use the macro `@i` to define a reversible function."

# ╔═╡ 70088425-6779-4a2d-ba6d-b0a34c8e93a6
@i function f1(x, y; constant)
	x += y * constant
end

# ╔═╡ f0e94247-f615-472b-8218-3fa287b38aa1
md"The function arguments contains keyword and non-keyword arguments. There is no `return` statement because this function returns input non-keyword variables automatically. `return` is both not needed and not allowed."

# ╔═╡ af738f89-3214-429c-9c7d-18a6ea0d9401
f1(1,2; constant=5)

# ╔═╡ 4d98d529-1e05-49be-9209-f0d9fcc206f7
md"The inverse call (or backward execution) is"

# ╔═╡ 48d7ebc1-5def-4a57-9ec1-3fc370a4543f
 (~f1)(11, 2; constant = 5)

# ╔═╡ e839547b-f47e-4a4e-b4d9-2c22921d80e4
md"Now we see the input arguments are restored"

# ╔═╡ 20145d75-004a-4c2f-b7ff-c400ca846d42
let
	content = md"""The above macro generates two functions, one is `f` and another is `~f` (or `Inv(f)`). The `x += y * constant` is translated to function `(x, y, constant) = PlusEq(*)(x, y, constant)`, where the function `PlusEq(*)` is bijective.

```julia
julia> using MacroTools

julia> MacroTools.prettify(@macroexpand @i function f(x, y; constant)
           x += y * constant
       end)
quote
    $(Expr(:meta, :doc))
    function $(Expr(:where, :(f(x, y; constant))))
        hare = wrap_tuple(((PlusEq)(*))(x, y, constant))
        x = hare[1]
        y = hare[2]
        constant = hare[3]
        (x, y)
    end
    if (NiLangCore)._typeof(f) != _typeof(~f)
        function $(Expr(:where, :((newt::_typeof(~f))(x, y; constant))))
            boar = wrap_tuple(((MinusEq)(*))(x, y, constant))
            x = boar[1]
            y = boar[2]
            constant = boar[3]
            (x, y)
        end
    end
end
```
"""
	HTML("<details><summary><strong>For experts</strong></summary>$(html(content))</details>")
end

# ╔═╡ b6dcd18c-606f-4340-b2ec-163e8bad03f5
title1("Variables")

# ╔═╡ a1a29f34-f8a9-4e9f-9afe-7d0096771440
title2("Allocate and deallocate a variable")

# ╔═╡ 90bd6ad4-3dd8-4e7c-b445-aed1d248a2ec
md"""
One can allocate a new variable `x` like `x ← constant` and deallocate a variable with known value with `x → known value`. They are reversible to each other with the following relation.

$(
twocol(md"
```julia
x ← constant
```", md"
```julia
x → constant
```
")
)

For example
"""

# ╔═╡ c0259e48-1973-486c-a828-1fcd3e4331c6
@i function f2a()
	tmp ← 1
	# some code that uses `tmp` for computing and restores it to `1`
	tmp → 1
end

# ╔═╡ 7bcc1bc6-5b2c-4aae-b71d-52482ae130e9
md"In reversible computing, it is forbidden to nullify (zero clear or assignment) the content in a variable."

# ╔═╡ fd5d65e7-3d83-45dd-87d7-ff1e1e769430
#`x` is in the input argument list, the content in `x` should not be cleared and re-allocated to another variable.
@test_throws LoadError macroexpand(NiLang, :(@i function f2a(x)
	x ← 1
	x → 1
end))

# ╔═╡ 4601df35-679f-465d-9191-c18748b2fd83
title2("Avoid shared read-write")

# ╔═╡ 4fc72b9d-19a2-40f1-a4a8-5e97d3d5e529
md"Shared read-write is not allowed"

# ╔═╡ a0fde16f-8454-4f5c-a29c-a9e415c0c311
# `y -= y` effectively clears the content in `y`, this is why shared read-write is so dangerous.
@test_throws LoadError macroexpand(NiLang, :(@i function f2b(y)
	y -= y
end))

# ╔═╡ 633ff8f3-8d93-4f73-bec2-c42070e6ece9
md"NiLang is more restrictive, it forbids shared read too. This is in purpose, shared read will become shared write to the gradient field in the back-propagation of gradients - the main goal of NiLang."

# ╔═╡ 57f2d890-b5a0-47b7-9e3d-af4d03b10605
# `shared read is also forbidden`
@test_throws LoadError macroexpand(NiLang, :(@i function f2b(x, y)
	x -= y * y  # should be written as `x -= y^2`
end))

# ╔═╡ 34063cd0-171e-46ce-80dd-52a341fa50a1
md"The correct way of avoiding shared read is renaming one of the variable."

# ╔═╡ 5d5d01db-8ff9-434c-8771-1fec6393e1fb
@i function f2c(x, y)
	tmp ← zero(y)
	tmp += y
	x -= y * tmp
	tmp -= y
	tmp → zero(y)
end

# ╔═╡ 10d85a50-f2f9-403e-8f6c-baef61cf702a
f2c(0.0, 3.0)

# ╔═╡ c830b274-b12b-46ee-98ad-a26d1447d491
title2("Read and write composite types, arrays et al.")

# ╔═╡ 398e5f66-bc22-4cf2-ae90-8cff600014f4
md"One can read and write the content of a composite type, "

# ╔═╡ dd8d370a-37d5-41de-8757-75f78db35dcb
@i function f2d(complex::Complex, array::AbstractArray, tuple::Tuple)
	# change the real part of a complex number
	complex.re += 1
	# change an array element
	array[2] += 1
	# change a tuple element is a bit sophisticated, `variable |> function` is not the same as the pipe operator in Julia and other functional programming languages. It means a "view" of a data that can be mutated. Here, we can not use the `[]` notation because, the front-end need to compile the code to the Julia language, however, the type information is not known in the macro expansion stage.
	(tuple |> tget(2)) += 1
end

# ╔═╡ 6c7f5858-d1e2-41ae-a422-4356a01b8632
f2d(1.0+1im, [1.0, 1.0], (1.0, 1.0))

# ╔═╡ f3503e6d-ded7-4797-9c2f-cb5a7d429949
md"#### Manipulating allocations on stack and dictionary"

# ╔═╡ 7ba86798-daba-49ef-a259-da4db353fdd8
md"""
The forward pass push a variable to the stack and zero-empty the variable, the backward pass. The backward pass checks the variable is empty and pop an variable from the stack and assign it to the variable.

$(
twocol(md"
```julia
PUSH!(stack, variable)
```", md"
```julia
POP!(stack, variable)
```
")
)


Or similarly, there is a `COPYPUSH!` instruction that copies the content in the variable to the stack without zero clearing the variable. In the backward pass, it asserts the target variable is the same as the popped variable.

$(
twocol(md"
```julia
COPYPUSH!(stack, variable)
```", md"
```julia
COPYPOP!(stack, variable)
```
")
)
"""

# ╔═╡ 6e88ede9-879e-42e9-9a7e-2aa3d4837e5d
@i function f2e(stack1::AbstractVector, stack2::AbstractVector)
	var1 ← 3.14
	var2 ← 2.73

	# copy a new variable to a stack
	COPYPUSH!(stack1, var1)
	# add a new variable to a stack and empty the original variable
	PUSH!(stack2, var2)
	
	# deallocate the original variables
	var2 → 0.0
	var1 → 3.14
end

# ╔═╡ 94cc42f7-6beb-453e-90cb-952d6f497a8e
f2e([], [])

# ╔═╡ 93936612-1447-4114-b864-aba43adef4bd
md"""
The forward pass of dictionary allocation adds an entry to the dictionary (cause key error if the key already exists), the backward pass. The backward pass checks the variable in the dictionary is consistent with the asserted variable, and delete the key.

$(
twocol(md"
```julia
dict[key] ← variable
```", md"
```julia
dict[key] → variable
```
")
)
"""

# ╔═╡ b2add70c-c5d6-4e0f-a153-43e21a197181
@i function f2f(dict::Dict)
	var1 ← 3.14

	# copy a new variable to a dict
	dict["data"] ← var1
	
	# deallocate the original variables
	var1 → dict["data"]
end

# ╔═╡ 5c03d5a5-99f0-4efd-9a32-ce6d7c2b266c
f2f(Dict())

# ╔═╡ 0863bd06-cc70-4dde-b3b2-0a466805a356
md"""$(title1("Integers, floating-point numbers, fixed-point numbers and logarithmic numbers"))

A fixed-point zero with 43 fraction bits can be declared as `x ← Fixed43(0.0)` or `x ← zero(Fixed43)`, a logarithmic one can be declared as `x ← ULogarithmic(1.0)`, `x ← one(ULogarithmic{Float64})` or `x ← ULogarithmic(Fixed43(1.0))`. A complex number zero can be defined as `x ← Complex(0.0, 0.0)`.


| Number Type | +=  | *= | ⊻= | Source |
| ---- | ---- | ---- | ---- | ---- |
| boolean | - |  - | ✓ | JuliaLang |
| integer | ✓ |  × | ✓ | JuliaLang |
| floating-point number | ✓ (rounding error) |  × | - | JuliaLang |
| fixed-point number | ✓ |  × | - | [FixedPointNumbers.jl](https://github.com/JuliaMath/FixedPointNumbers.jl)
| logarithmic number | × |  ✓ | - |  [LogarithmicNumbers.jl](https://github.com/cjdoris/LogarithmicNumbers.jl)

* `✓`: an operation has its reverse when operating on a number type.
* `×`: an operation does not have a reverse when operating on a number type.
* `-`: an operation does not apply on a number type.

The `+=` operation is not regoriously reversible on floating point numbers, but we ignore the rounding errors in NiLang and use the reversibility check to detect the potential too-large rounding errors. Whether logarithmic number has rounding errors depends on its content type. If it uses floating point numbers as storage, then yes, otherwise if it uses fixed point number as the content type, then no.

One can use the `y ⊙= convert(x)` statement to convert `x` to the target type `typeof(y)` and accumulate it to `y`. Here `⊙=` can be one of `+=`, `*=` and `⊻=` that has its reverse on type `typeof(y)`.
"""

# ╔═╡ 77947e00-42c3-4c9e-b62a-b4b29489db43
@i function f3a(y1::ULogarithmic{T}, x::ULogarithmic{T}) where T
	y1 += x
end; @test_throws MethodError f3a(ULogarithmic(1.0), ULogarithmic(3.0))

# ╔═╡ 2614127d-34fb-4c3d-b678-42693f3c9341
@i function f3b(y1::ULogarithmic{T}, x::ULogarithmic{T}) where T
	y1 *= x
end; @test f3b(ULogarithmic(1.0), ULogarithmic(3.0))[1] == ULogarithmic(3.0)

# ╔═╡ 20d6e8a0-2cf5-48ad-9549-60506b42b970
@i function f3c(y1::ULogarithmic{T}, x::ULogarithmic{T}) where T
	y1 += x
end

# ╔═╡ 7dee5748-ed73-4e13-aa80-7a50efbc8449
@i function f3d(y1::Fixed43, x::Fixed43) where T
	y1 *= x
end; @test_throws MethodError f3d(Fixed43(1.0), Fixed43(2.0))

# ╔═╡ 8f169235-3bd1-4cc4-a083-79736d306ad5
example("computing x ^ 3 with logarithmic numbers")

# ╔═╡ dfc9d305-5bce-4555-bfa3-d8d61fe4ca09
@i function f3e(y::ULogarithmic{T}, x::T) where T
	for i=1:3
		y *= convert(x)
	end
end; @test f3e(ULogarithmic(1.0), 3.0)[1] ≈ 27

# ╔═╡ c4cd9f88-9cd6-4364-b016-78f90aba6a66
md"""## Mathematical Operations and Elementary Functions
A mathematical operation can be defined in the form `⊙=`, where `⊙` can be `+`, `-`, `*`, `/` or `⊻`. See section $(titleref("Integers, floating-point numbers, fixed-point numbers and logarithmic numbers")) for their supported number types.

When we say an elementary function is supported, we mean its diffrule is defined. A list of supported elementary differentiable functions are

| instruction | output   |
| ----------- | ---------- |
| ``{\rm FLIP}(y)`` | ``\sim y`` |
| ``{\rm NEG}(y)`` | ``-y`` |
| ``{\rm INC}(y)`` | ``y+1`` |
| ``{\rm DEC}(y)`` | ``y-1`` |
| ``{\rm INV}(y)`` | ``y ^ {-1}`` |
| ``{\rm HADAMARD}(x, y)`` | ``\frac{1}{\sqrt{2}}(x+y), \frac{1}{\sqrt{2}}(x-y)``
| ``{\rm SWAP}(a, b)`` | ``b, a`` |
| ``{\rm ROT}(a, b, \theta)`` | ``a \cos\theta - b\sin\theta, b \cos\theta + a\sin\theta, \theta`` |
| ``{\rm IROT}(a, b, \theta)`` | ``a \cos\theta + b\sin\theta, b \cos\theta - a\sin\theta, \theta`` |
| ``y \mathrel{\{+,-\}}= f_{+-}(args...)`` | ``y\{+, -\}f_{+-}(args...), args...`` |
| ``y \mathrel{\{*, /\}}= f_{*/}(args...)`` | ``y\{*, /\}f_{*/}(args...), args...`` |

"." is the broadcasting operations in Julia. Functions ``f_{+-} ∈ \rm \{identity, +, -, *, /, ^\wedge, abs, abs2,``
``\rm sqrt, exp, log, sin, sinh, asin, cos, cosh, acos, tan, tanh, atan, sincos, convert\}`` and ``f_{*/}∈\rm \{identity, +, -, *, /, ^\wedge, convert\}.`` 
Functions `FLIP`, `NEG`, `INV`, `HADAMARD`, `SWAP` and `y ⊻= f_{⊻}(args...)` are self-reversible (or reflexive). {`ROT`, `IROT`} and {`INC`, `DEC`}, {`y += f_{+-}(args...)`, `y -= f_{+-}(args...)`} and {`y *= f_{*/}(args...)`, `y /= f_{*/}(args...)`} are pair-wise reversible.

For Jacobians and Hessians defined on these instructions, please check this [blog post](https://giggleliu.github.io/2020/01/18/jacobians.html).
"""
# They are translated to `y += f(args...)` is translated to `PlusEq(f)(y, args...)`, `y -= f(args...)` is translated to `MinusEq(f)(y, args...)`, `y *= f(args...)` is translated to `MulEq(f)(y, args...)`, `y /= f(args...)` is translated to `DivEq(f)(y, args...)` and `y ⊻= f(args...)` is translated to `XorEq(f)(y, args...)`.

# ╔═╡ 648cdcd6-f4f5-461f-a525-4b350cae9eb0
example("defining an elementary function")

# ╔═╡ f502b8c1-9b80-4e67-80e8-a64ddb88fb0f
@test (~M.new_forward)(M.new_forward(3.0)) ≈ 3.0

# ╔═╡ fd9cf757-2698-4886-9f0a-c6c23ff0d331
@test NiLang.AD.check_grad(M.new_forward, (3.0,); iloss=1)

# ╔═╡ dda6652a-d063-4511-8041-e869bb88ca26
@test NiLang.AD.check_grad(M.new_backward, (3.0,); iloss=1)

# ╔═╡ f6bfa015-c101-45e8-995c-2bb6a3b7dc7d
title1("Types and arrays")

# ╔═╡ 8651d7ec-6bcd-4dbe-a062-c4bde32e5e91
md"NiLang is compatible with Julia's type system,
a type can be accessed in NiLang as long as its default constructor is not overloaded, because NiLang requires the default constructor to \"modify\" a field of a immutable type. For example, one can modify the real and imaginary part of a complex number directly in NiLang."

# ╔═╡ 9f5f9de3-9558-4c18-9d98-b77d19b570ec
example("Complex valued log")

# ╔═╡ 6dfcfa19-f78f-4dac-89f7-d3c5dbe17987
@i function complex_log(y!::Complex{T}, x::Complex{T}) where T
    n ← zero(T)
    n += abs(x)

    y!.re += log(n)
    y!.im += angle(x)

    n -= abs(x)
    n → zero(T)
end; @test complex_log(0.0im, 3.0+2.0im)[1] ≈ log(3.0+2.0im)

# ╔═╡ 99d6fe7b-d704-48f3-b115-2b3159a78068
md"One can also modify the `Array` directly"

# ╔═╡ 1950ff70-54eb-4ece-a26d-a23fd0e90f5a
@i function arrayaxpy!(y!::Vector{T}, a::T, x::Vector{T}) where T
    for i=1:length(x)
		y![i] += a * x[i]
	end
end; @test arrayaxpy!(zeros(10), 2.0, collect(1.0:10.0))[1] ≈ collect(2.0:2.0:20.0)

# ╔═╡ 21458f81-9007-46f8-92e0-7a17c60beb36
md"To modify an element of a `Tuple`, we need to use a different style to avoid confusion with array"

# ╔═╡ 7813f4ce-6e98-45f3-94a8-7f5981129f2b
@i function tupleaxpy!(y!::NTuple{N,T}, a::T, x::NTuple{N,T}) where {N, T}
    for i=1:length(x)
		(y! |> tget(i)) += a * (x |> tget(i))
	end
end; @test tupleaxpy!((0,0,0), 2, (1,2,3))[1] == (2, 4, 6)

# ╔═╡ 59ec7cb7-6011-456d-9f57-a55bb8ea51a0
md"Here `data |> function` is called a *dataview* of an object, it defines a modifiable view for a field of a data."

# ╔═╡ d6519029-231d-4c63-b47d-684462dab287
md"""
## Functions

One can still define inline functions like
```julia
@i @inline function f(args...)
	...
end
```

However, the generated functions are not yet supported.
"""

# ╔═╡ aacf63a2-9708-40db-8928-049621a7bbc4
md"## Control flows"

# ╔═╡ ad0097e7-c8ad-457a-82a9-18b998a9e9fb
md"""
#### If statement

The condition expression in `if` statement contains two parts, one is precondition and another is postcondition.

$(
twocol(md"
```julia
if (precondition, postcondition)
	...
end
```
", md"
```julia
if (postcondition, precondition)
	~(...)
end
```
")
)

where `...` are statements and `~(...)` are the backward execution of them.
"""

# ╔═╡ 4c03cde9-b643-40ff-b275-f1795f88949e
title2("For statement")

# ╔═╡ fae7c74e-d25e-4c1e-ac97-199e6dae3365
md"""
The reversible `for` statement is similar to its irreversible counterpart.

$(
twocol(md"
```julia
for iter = start:step:stop
	...
end
```
", md"
```julia
for iter = stop:-step:start
	~(...)
end
```
")
)
"""

# ╔═╡ 95c41bd1-e50a-42e8-93c3-3b754a458c13
title2("While statement")

# ╔═╡ b8629aeb-6c9a-44ed-87a1-9ab22d9485ed
md"""
The reversible `while` statement starts with the `@from` macro.

$(
twocol(md"
```julia
@from condition1 while condition2
	...
end
```
", md"
```julia
@from !(condition2) while !(condition1)
	...
end
```
")
)
"""

# ╔═╡ 3b211406-041f-4b41-acae-3958e4a37224
md"where the `condition1` in the forward pass is a condition that holds before entering the loop body, but broken at the first iteration, while `condition2` is just a normal while condition. In the backward pass, `!(condition1)` becomes the criteria to break the loop."

# ╔═╡ 62522772-cb59-4d13-acdd-d5067b223910
@i function f7(x, i)
	@from i==0 while i<10
		i += 1
		x += 1
	end
end

# ╔═╡ 2ba68a0f-6e36-4ea2-a91d-6af43741bad1
f7(1, 0)

# ╔═╡ 75cebaf1-38de-475f-892e-346fd2b46f6f
@test (~f7)(11, 10) == (1, 0)

# ╔═╡ 72dcf2fe-eb48-4dee-8121-efafc87637e3
title2("Compute-copy-uncompute statement")

# ╔═╡ 84321198-93d4-4d22-8c0f-a5a10b884e1f
md"The *compute-copy-uncompute* statement is a widely used design pattern in reversible programming. 
We compute the forward pass for the result, then we copy the result to the output variable, and run the backward pass to erase intermediate results.
For example, to compute `y = x * exp(k)`, we might write the following code"

# ╔═╡ 32244789-afbf-4215-97cd-15483f438eee
@i function f6a(y, x, k)
	expk ← zero(k)
	expk += exp(k)
	y += x * expk
	# uncompute the ancilla and deallocate it
	expk -= exp(k)
	expk → zero(k)
end; @test f6a(0.0, 2.0, 3.0)[1] ≈ 2.0 * exp(3.0)

# ╔═╡ 4b7f0baf-0316-4da7-9ded-50c064ddbaa3
md"It is equivalent to the following statement that generates the backward pass automatically for you."

# ╔═╡ 0e02952c-7589-4606-b006-16a9f3e52ae1
@i function f6b(y, x, k)
	# record the forward pass
	@routine begin
		expk ← zero(k)
		expk += exp(k)
	end
	y += x * expk
	# reverse execute the recorded the program
	~@routine
end; @test f6b(0.0, 2.0, 3.0)[1] ≈ 2.0 * exp(3.0)

# ╔═╡ f0904d3f-1bf1-459c-9959-b53c0f774e3f
example("Computing Fibonacci numbers")

# ╔═╡ 19bb2af5-2a67-453d-82b0-7d3059b1fa47
md"The sequence of Fibonacci numbers are: 1, 1, 2, 3, 5, 8"

# ╔═╡ 5b5858bf-63ac-4e31-a516-055a9cd18ffe
@i function rfib(out!, n::T) where T
    n1 ← zero(T)
    n2 ← zero(T)
    @routine begin
        n1 += n - 1
        n2 += n - 2
    end
    if (value(n) <= 2, ~)
        out! += 1
    else
        rfib(out!, n1)
        rfib(out!, n2)
    end
    ~@routine
end

# ╔═╡ 95060588-f24b-4eeb-9b0b-ed7159962a3c
@test rfib(0, 6)[1] == 8

# ╔═╡ a7d47e83-7f44-49d0-a43d-e01316fc6eba
title1("Performance Tips")

# ╔═╡ 45985244-adbf-4d6d-9732-a963cca62212
title2("Remove reversibility check")

# ╔═╡ 83d7e75f-7273-4c6a-bec1-a2180ebc3fb9
md"This can be done by putting an `@invcheckoff` before a code block."

# ╔═╡ 7fb05c65-f47c-430a-b588-c2f9bade40a9
example("computing the exp function by Taylor expansion")

# ╔═╡ 14c0caa1-51ea-448c-a7dc-d06e34dd0895
md"Note: this is not a clever implementation. There is an approach of defining it without allocation."

# ╔═╡ 457d07bb-e999-413e-8f29-58714670296f
@i function exp_with_reversibility_check(y::T, x::T) where T
	@routine begin
		N ← 1_000
		anc ← zeros(T, N)
		anc[1] += 1
		anc_y ← T(2.0)
		for i=2:N
			@routine begin
				temp ← zero(T)
				temp += x * anc[i-1]
			end
			anc[i] += temp / i
			anc_y += anc[i]
			~@routine
		end
	end
	y += anc_y
	~@routine
end

# ╔═╡ ac53eac0-1a59-4407-8bf6-3d8b966a9bff
@benchmark exp_with_reversibility_check(0.0, 1.0) seconds=0.3

# ╔═╡ 85c8ac7b-54f5-47dc-bd50-e78ffd6cf1cf
@i function exp_without_reversibility_check(y::T, x::T) where T
	@routine @invcheckoff begin
		N ← 1_000
		anc ← zeros(T, N)
		anc[1] += 1
		anc_y ← T(2.0)
		for i=2:N
			@routine begin
				temp ← zero(T)
				temp += x * anc[i-1]
			end
			anc[i] += temp / i
			anc_y += anc[i]
			~@routine
		end
	end
	y += anc_y
	~@routine
end

# ╔═╡ 95c55847-0591-4f7f-b9a1-aa974ccfef69
@benchmark exp_without_reversibility_check(0.0, 1.0) seconds=0.3

# ╔═╡ Cell order:
# ╠═2061b434-0ad1-46eb-a0c7-1a5f432bfa62
# ╠═a4e76427-f051-4b29-915a-fdfce3a299bb
# ╟─c2c7b4d4-f8c9-4ebf-8da2-0103f03136e7
# ╟─605872cf-f3fd-462e-a2b1-7d1c5ae45efd
# ╟─fb3dee44-5fa9-4773-8b7f-a83c44358545
# ╠═d941d6c2-55bf-11eb-0002-35c7474e4050
# ╠═70088425-6779-4a2d-ba6d-b0a34c8e93a6
# ╟─f0e94247-f615-472b-8218-3fa287b38aa1
# ╠═af738f89-3214-429c-9c7d-18a6ea0d9401
# ╟─4d98d529-1e05-49be-9209-f0d9fcc206f7
# ╠═48d7ebc1-5def-4a57-9ec1-3fc370a4543f
# ╟─e839547b-f47e-4a4e-b4d9-2c22921d80e4
# ╟─20145d75-004a-4c2f-b7ff-c400ca846d42
# ╟─b6dcd18c-606f-4340-b2ec-163e8bad03f5
# ╟─a1a29f34-f8a9-4e9f-9afe-7d0096771440
# ╟─90bd6ad4-3dd8-4e7c-b445-aed1d248a2ec
# ╠═c0259e48-1973-486c-a828-1fcd3e4331c6
# ╟─7bcc1bc6-5b2c-4aae-b71d-52482ae130e9
# ╠═fd5d65e7-3d83-45dd-87d7-ff1e1e769430
# ╟─4601df35-679f-465d-9191-c18748b2fd83
# ╟─4fc72b9d-19a2-40f1-a4a8-5e97d3d5e529
# ╠═a0fde16f-8454-4f5c-a29c-a9e415c0c311
# ╟─633ff8f3-8d93-4f73-bec2-c42070e6ece9
# ╠═57f2d890-b5a0-47b7-9e3d-af4d03b10605
# ╟─34063cd0-171e-46ce-80dd-52a341fa50a1
# ╠═5d5d01db-8ff9-434c-8771-1fec6393e1fb
# ╠═10d85a50-f2f9-403e-8f6c-baef61cf702a
# ╟─c830b274-b12b-46ee-98ad-a26d1447d491
# ╟─398e5f66-bc22-4cf2-ae90-8cff600014f4
# ╠═dd8d370a-37d5-41de-8757-75f78db35dcb
# ╠═6c7f5858-d1e2-41ae-a422-4356a01b8632
# ╟─f3503e6d-ded7-4797-9c2f-cb5a7d429949
# ╟─7ba86798-daba-49ef-a259-da4db353fdd8
# ╠═6e88ede9-879e-42e9-9a7e-2aa3d4837e5d
# ╠═94cc42f7-6beb-453e-90cb-952d6f497a8e
# ╟─93936612-1447-4114-b864-aba43adef4bd
# ╠═b2add70c-c5d6-4e0f-a153-43e21a197181
# ╠═5c03d5a5-99f0-4efd-9a32-ce6d7c2b266c
# ╟─0863bd06-cc70-4dde-b3b2-0a466805a356
# ╠═77947e00-42c3-4c9e-b62a-b4b29489db43
# ╠═2614127d-34fb-4c3d-b678-42693f3c9341
# ╠═20d6e8a0-2cf5-48ad-9549-60506b42b970
# ╠═7dee5748-ed73-4e13-aa80-7a50efbc8449
# ╟─8f169235-3bd1-4cc4-a083-79736d306ad5
# ╠═dfc9d305-5bce-4555-bfa3-d8d61fe4ca09
# ╟─c4cd9f88-9cd6-4364-b016-78f90aba6a66
# ╟─648cdcd6-f4f5-461f-a525-4b350cae9eb0
# ╠═8c2c4fa6-172f-4dde-a279-5d0aecfdbe46
# ╠═f502b8c1-9b80-4e67-80e8-a64ddb88fb0f
# ╠═fd9cf757-2698-4886-9f0a-c6c23ff0d331
# ╠═dda6652a-d063-4511-8041-e869bb88ca26
# ╟─f6bfa015-c101-45e8-995c-2bb6a3b7dc7d
# ╟─8651d7ec-6bcd-4dbe-a062-c4bde32e5e91
# ╟─9f5f9de3-9558-4c18-9d98-b77d19b570ec
# ╠═6dfcfa19-f78f-4dac-89f7-d3c5dbe17987
# ╟─99d6fe7b-d704-48f3-b115-2b3159a78068
# ╠═1950ff70-54eb-4ece-a26d-a23fd0e90f5a
# ╟─21458f81-9007-46f8-92e0-7a17c60beb36
# ╠═7813f4ce-6e98-45f3-94a8-7f5981129f2b
# ╟─59ec7cb7-6011-456d-9f57-a55bb8ea51a0
# ╠═d6519029-231d-4c63-b47d-684462dab287
# ╟─aacf63a2-9708-40db-8928-049621a7bbc4
# ╟─ad0097e7-c8ad-457a-82a9-18b998a9e9fb
# ╟─4c03cde9-b643-40ff-b275-f1795f88949e
# ╟─fae7c74e-d25e-4c1e-ac97-199e6dae3365
# ╟─95c41bd1-e50a-42e8-93c3-3b754a458c13
# ╟─b8629aeb-6c9a-44ed-87a1-9ab22d9485ed
# ╟─3b211406-041f-4b41-acae-3958e4a37224
# ╠═62522772-cb59-4d13-acdd-d5067b223910
# ╠═2ba68a0f-6e36-4ea2-a91d-6af43741bad1
# ╠═75cebaf1-38de-475f-892e-346fd2b46f6f
# ╟─72dcf2fe-eb48-4dee-8121-efafc87637e3
# ╠═84321198-93d4-4d22-8c0f-a5a10b884e1f
# ╠═32244789-afbf-4215-97cd-15483f438eee
# ╟─4b7f0baf-0316-4da7-9ded-50c064ddbaa3
# ╠═0e02952c-7589-4606-b006-16a9f3e52ae1
# ╟─f0904d3f-1bf1-459c-9959-b53c0f774e3f
# ╟─19bb2af5-2a67-453d-82b0-7d3059b1fa47
# ╠═5b5858bf-63ac-4e31-a516-055a9cd18ffe
# ╠═95060588-f24b-4eeb-9b0b-ed7159962a3c
# ╟─a7d47e83-7f44-49d0-a43d-e01316fc6eba
# ╟─45985244-adbf-4d6d-9732-a963cca62212
# ╟─83d7e75f-7273-4c6a-bec1-a2180ebc3fb9
# ╟─7fb05c65-f47c-430a-b588-c2f9bade40a9
# ╟─14c0caa1-51ea-448c-a7dc-d06e34dd0895
# ╠═457d07bb-e999-413e-8f29-58714670296f
# ╠═3199a048-7b39-40f8-8183-6a54cccd91b6
# ╠═ac53eac0-1a59-4407-8bf6-3d8b966a9bff
# ╠═85c8ac7b-54f5-47dc-bd50-e78ffd6cf1cf
# ╠═95c55847-0591-4f7f-b9a1-aa974ccfef69

### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ d941d6c2-55bf-11eb-0002-35c7474e4050
using NiLang, Test

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
	using PlutoUI: TableOfContents
	using Pkg
	pkgversion(m::Module) = Pkg.TOML.parsefile(NiLang.project_relative_path("Project.toml"))["version"]
end;

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

# The following is need only when your function is differentiable
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

# ╔═╡ 0e1ba158-a6bc-401c-9ba7-ed78020ad068
using Base.Threads

# ╔═╡ 3199a048-7b39-40f8-8183-6a54cccd91b6
using BenchmarkTools

# ╔═╡ a4e76427-f051-4b29-915a-fdfce3a299bb
html"""
<div align="center">
<a class="Header-link " href="https://github.com/GiggleLiu/NiLang.jl" data-hotkey="g d" aria-label="Homepage " data-ga-click="Header, go to dashboard, icon:logo">
  <svg class="octicon octicon-mark-github v-align-middle" height="32" viewBox="0 0 16 16" version="1.1" width="32" aria-hidden="true"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
</a>
<br>
<a href="https://raw.githubusercontent.com/GiggleLiu/NiLang.jl/master/notebooks/documentation.jl" target="_blank" download>Download this notebook</a>
</div>
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
md"# NiLang's (v$(pkgversion(NiLang))) Documentation

NiLang is a embeded domain specific language (eDSL) in Julia, so one need to install [Julia](https://julialang.org/) first. Before reading this documentation, you need to know basic Julia grammar, and how to install and use packages.
Also, it might be good to read the [README](https://github.com/GiggleLiu/NiLang.jl) first. In this tutorial, we focus on

* NiLang grammar and design patterns.
* Automatic differentiation based on reversible programming
"

# ╔═╡ 611b577f-4722-42bf-8f8e-aeb2fb30be71
md"""
|  symbol  |  meaning  | how to type |
| -------  | --------- | ----------- |
| ← | allocate |  \leftarrow + TAB |
| → | deallocate | \rightarrow + TAB |
| ↔ | exchange | \leftrightarrow + TAB |
| ∅ | empty variable | \emptyset + TAB |
| ~ | inverse | ~ |
"""

# ╔═╡ 605872cf-f3fd-462e-a2b1-7d1c5ae45efd
title1("Getting started")

# ╔═╡ fb3dee44-5fa9-4773-8b7f-a83c44358545
md"
After installing NiLang in a Julia REPL by typing `]add NiLang`, one can use NiLang and use the macro `@i` to define a reversible function ``f_1: (x, y) → (x+5y, y)``."

# ╔═╡ 70088425-6779-4a2d-ba6d-b0a34c8e93a6
@i function f1(x, y; constant)
	x += y * constant
end

# ╔═╡ f0e94247-f615-472b-8218-3fa287b38aa1
md"The function arguments contains keyword and non-keyword arguments, where non-keyword arguments can only be used as constants. There is no `return` statement because this function returns input non-keyword variables automatically. `return` is both not needed and not allowed."

# ╔═╡ 2581aa33-1dc5-40b1-aa9f-6a11cc750c93
md"In NiLang, a variable is mutable. After running an instruction that defines the mapping ``\mathbb{R}^n\rightarrow\mathbb{R}^n`` (notice inputs and outputs have the same shapes), the outputs are directly assigned back. In other words, every instruction changes the variable inplace."

# ╔═╡ af738f89-3214-429c-9c7d-18a6ea0d9401
f1(1.0, 2.0; constant=5.0)

# ╔═╡ 4d98d529-1e05-49be-9209-f0d9fcc206f7
md"The inverse call (or backward execution) is"

# ╔═╡ 48d7ebc1-5def-4a57-9ec1-3fc370a4543f
 (~f1)(11.0, 2.0; constant = 5.0)

# ╔═╡ e839547b-f47e-4a4e-b4d9-2c22921d80e4
md"Now we see the input arguments are restored"

# ╔═╡ 60575978-081a-4bca-a3ed-2b51cd6abc92
md"One can also differentiating the function. This is a two-in two-out function (the keyword arguments are not included), we need to specify which variable is the loss when asking for the gradients."

# ╔═╡ f98305cb-4ba2-404a-a5c3-65510e059504
NiLang.AD.gradient(f1, (1.0, 2.0); iloss=1, constant=5.0)

# ╔═╡ e8cd6667-597f-458b-8465-1822e09a7891
md"Here, we specify the first variable as the one that stores the loss. We get
```math
\begin{cases}
\frac{\partial x+5y}{\partial x}=1\\
\frac{\partial x+5y}{\partial y}=5\\
\end{cases}
```
"

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
	HTML("<details><summary><strong>If you know macros</strong></summary>$(html(content))</details>")
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
@i function alloc_func1()
	tmp ← 1
	# some code that uses `tmp` for computing and restores it to `1`
	tmp → 1
end

# ╔═╡ 7bcc1bc6-5b2c-4aae-b71d-52482ae130e9
md"It is forbidden to repeatedly allocate a variable, use a deallocate variable or repeatedly deallocate a variable."

# ╔═╡ fd5d65e7-3d83-45dd-87d7-ff1e1e769430
#`x` is in the input argument list, the content in `x` should not be cleared and re-allocated to another variable.
@test_throws LoadError macroexpand(NiLang, :(@i function alloc_func1(x)
	x ← 1
	x → 1
end))

# ╔═╡ 8bbffa31-04a6-49ca-b36f-4d4140d75992
md"allocate multiple variable of the same type at one time"

# ╔═╡ a6f18c34-80ee-4b52-9ff8-f3c1b1d80f90
@i function power12(y, x::T) where T
	@zeros T a b c  # three variable of type `T`
	a += x^2
	b += a^2
	c += b*a
	y += c^2
	c -= b*a
	b -= a^2
	a -= x^2
	@safe @show a b c x y
	~@zeros T a b c
end

# ╔═╡ a694132b-4f52-467f-8bc4-dc32fe2812db
@test power12(0, 2)[1] == 4096

# ╔═╡ 8c2c82f2-1240-4f2f-830e-ee8021c1a41a
md"One can copy and push a value into a stack and use it later. It inverse operation will pop out a variable and assert its value."

# ╔═╡ 6203cf10-f8cc-4fb9-b814-7552b68c01dc
twocol(md"
```julia
stack[end+1] ← variable
```", md"
```julia
stack[end] → variable
```
")

# ╔═╡ f97a6bab-b9f9-4b95-98a9-381c51397526
@i function stack_push_and_pop!(stack, x, y, z)
	z += y
	stack[end+1] ← x  # copy a variable into a stack
	stack[end+1] ← y 
	stack[end] → z  # pop a variable from a stack, `z` must have the same value as the variable.
end

# ╔═╡ 2a2970f4-ab01-486b-89a2-6ff96f734018
md"A less recommended approach is using the global stacks in NiLang, since NiLang is an eDSL, it can not guarantee the access order. Available global stacks are `FLOAT64_STACK`, `COMPLEXF64_STACK`, `INT64_STACK` and their 32 bit counter parts, as well as a `BOOL_STACK`."

# ╔═╡ 0b80d9be-53d7-4bf3-a558-659607af4709
@i function stack_push_and_pop!(x, y)
	GLOBAL_STACK[end+1] ← x  # copy a variable into a stack
	FLOAT64_STACK[end+1] ← y 
end

# ╔═╡ 4ca48a2e-43da-457a-8e9f-6476097e4d7b
let
	stack = FastStack{Float64}(1000) # a preallocated stack of size 1000
	stack_push_and_pop!(stack, 5.0, 1.0, 0.0)  # you will get a stack of size 1
	@test length(stack) == 1
end

# ╔═╡ 92362fda-bae2-4e35-bfe4-dcaea853d50b
let
	NiLang.empty_global_stacks!()  # empty stacks
	stack_push_and_pop!(5.0, 1.0)
	@test length(GLOBAL_STACK) == 1
	@test length(FLOAT64_STACK) == 1
end

# ╔═╡ db9e7940-39f1-4ccf-ac70-146a521daa6e
md"one can also allocate and deallocate on dicts"

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

# ╔═╡ 2349e3ea-3053-42a4-b9d9-f97a76e4abd7
title2("Exchange two variables")

# ╔═╡ 269f18ee-3cd8-466a-a522-7c624503e31b
let 
	expr = twocol(md"
```julia
var1 ↔ var2
```", md"
```julia
var1 ↔ var2
```
")
md"""One can exchange two variables is using `↔`.
$expr
"""
end

# ╔═╡ 89139719-c478-4066-9452-f9893f36d561
@i function exchange_func1(x, y)
	x ↔ y
end

# ╔═╡ d620c5ee-7d9c-4d3f-9e87-0c828dfab9ca
exchange_func1(3, 5)

# ╔═╡ 255d01b9-a873-4e63-9298-9d8f073348b0
md"One can also make a \"link\" by exchanging a variable with an empty variable such as `var::∅` and `stack[end+1]`. The forward pass push a variable to the stack and deallocate variable. The backward pass pops a variable and asserts its value."

# ╔═╡ f6cf1729-766c-4ed7-b004-c8c8ec6c7e07
let 
	expr = twocol(md"
```julia
stack[end+1] ↔ var2
var1::∅ ↔ var2
```", md"
```julia
stack[end] ↔ var2::∅
var1 ↔ var2::∅
```
")
end


# ╔═╡ 3645d672-423f-4ac8-805f-0452793fee5a
@i function exchange_func2(x, y)
	anc ← 0.0
	anc += x * y
	anc ↔ z::∅  # declare `z` as an empty variable
	# after exchange, `anc` is empty and deallocated automatically.
	z -= x * y
	z → 0.0
end

# ╔═╡ c2a0024e-11dd-4ef7-8346-4374d98cafc0
exchange_func2(3, 4)

# ╔═╡ b20004e9-3c73-4dfb-8fd5-f377786fd53b
md"When exchanging with a stack top + 1, it means push and deallocate."

# ╔═╡ 5c1952b1-5016-4c87-b23c-8e6a235bf8cd
@i function stack_exchange(stack, y, x)
	stack[end+1] ↔ y  # push a variable into a stack and deallocate `y`
	y ← 1.2  # since `y` is deallocated, you can assign any value to it
	stack[end] ↔ anc::∅  # pop a variable to `anc`
	anc ↔ x    # exchange `anc` and x
	stack[end+1] ↔ anc  # push `anc` back to stack
end

# ╔═╡ 8e4470ee-01da-4547-b091-c4f65cd729b0
let
	stack = FastStack{Float64}(1000) # a preallocated stack
	stack_exchange(stack, 2.0, 3.0)  # you will get a stack of size 2
	@test length(stack) == 1 && stack.data[1] == 3.0
end

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

# ╔═╡ b52648bf-a28a-48af-8912-31729d943ce0
md"Shared read-write issue is more tricky when one uses NiLang to write kernel function in a parallel program (multi-threading, MPI and CUDA). See $(titleref(\"Multi-threading and CUDA\")) for details."

# ╔═╡ f45db10f-a836-40f3-9d8d-054ea6540e87
title2("Protect special variables")

# ╔═╡ 1903563e-ccc2-44d9-9dbe-e5dede275b3c
md"""
By default, NiLang assigns the value back to the variable after a call. Sometime it can cause issues for special variables like function, type parameters and constants (including the results generated from a function call). One can use `@const` (assert a variable is a constant) or `@skip!` (skip assigning back) to avoid such complications.
"""

# ╔═╡ 7ec577d4-1e19-484e-ac1f-5da3f47d1ec4
md"""
###### A function variable
"""

# ╔═╡ 583f2585-15a3-47c6-a70e-e2f002754028
md"When using a function (e.g. `exp` as shown bellow) as an variable, one should be careful about the scope issue."

# ╔═╡ 60e6ff80-3593-4ae4-a273-914847f692db
@i function f2g(y, f, x)
	y += f(x)
end

# ╔═╡ 9e5cfd68-b58d-4d83-aae2-447e5f805c97
@i function f2h(y, x)
	f2g(y, exp, x)
end

# ╔═╡ dc85a942-cf52-4405-ad03-32a768e1b6e7
@test_throws UndefVarError f2h(0.0, 3.0)

# ╔═╡ 5cdd346b-10a5-485c-ba78-4c0b3cb0e02f
md"We see an error, but why calling `f2g` causes an error? If one check the generated code with `macroexpand`, one will see the `exp` is assigned in the local scope. The compiler takes it as a local variable and compaints that `exp` is not defined."

# ╔═╡ af9287b7-6131-46f6-beb8-6885e55e1975
macroexpand(NiLang, :(@i function f2h(y, x)
	f2g(y, exp, x)
end)) |> NiLangCore.rmlines

# ╔═╡ e20eeabf-1c80-431e-8cfc-4d1b79c52b5a
@i function f2i(y, x)
	f2g(y, (@const exp), x)
end; @test f2i(0.0, 3.0)[1] == exp(3)

# ╔═╡ e45d53b9-6f02-4293-ab4f-85bd751993ad
md"""
###### Type parameters and functions
"""

# ╔═╡ 90d30eea-53de-48a0-9700-ff35681fdf38
md"Type parameter can not be assigned back too."

# ╔═╡ 390f58a5-6f5f-4d3a-bb16-ba04e43a07e7
@test_throws ErrorException Core.eval(NiLang, :(@i function f2j(t::Type{T}, x) where T
	x += one(T)
end))

# ╔═╡ 2b57443e-a516-434b-be86-80616a98e2f5
@i function f2k(t::Type{T}, x) where T
	x += one(@skip! T)
end; @test f2k(Float64, 0.0)[2] == 1.0

# ╔═╡ fc2e27f9-b7ba-44cc-a953-6745548ad733
md"A function call that returning a constant should also be decorated with the `@const` (assert a variable is not changed) or `@skip!` (do not assign this variable back) macro."

# ╔═╡ fc744931-360b-4478-9f77-c50f048de243
@test_throws LoadError macroexpand(NiLang, :(@i function f2l(y, x::Matrix) where T
	y += size(x, 1) * size(x, 2)
end))

# ╔═╡ 9a152b36-f377-44da-9700-ca9e05e365ff
@i function f2m(y, x::Matrix) where T
	y += (@const size(x, 1)) * (@const size(x, 2))
end; @test f2m(0, randn(3,4))[1] == 12

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

# ╔═╡ a0bae195-04e1-4642-9e14-fe4691e0906b
md"Fixed point numbers and Floating point numbers are reversible under the `+=` operation"

# ╔═╡ 20d6e8a0-2cf5-48ad-9549-60506b42b970
@i function f3c(y1::T, x::T) where T<:Union{Fixed43, AbstractFloat}
	y1 += x
end; @test f3c(Fixed43(0.5), Fixed43(0.6))[1] === Fixed43(1.1) && f3c(0.5, 0.6)[1] === 1.1

# ╔═╡ 7dee5748-ed73-4e13-aa80-7a50efbc8449
@i function f3d(y1::T, x::T) where T<:Union{Fixed43, AbstractFloat}
	y1 *= x
end; @test_throws MethodError f3d(Fixed43(1.0), Fixed43(2.0))

# ╔═╡ 4c719e9b-641e-404e-9ab7-59e89135f3ba
md"Logarithmic numbers are reversible under the `*=` operation"

# ╔═╡ 77947e00-42c3-4c9e-b62a-b4b29489db43
@i function f3a(y1::ULogarithmic{T}, x::ULogarithmic{T}) where T
	y1 += x
end; @test_throws MethodError f3a(ULogarithmic(1.0), ULogarithmic(3.0))

# ╔═╡ 2614127d-34fb-4c3d-b678-42693f3c9341
@i function f3b(y1::ULogarithmic{T}, x::ULogarithmic{T}) where T
	y1 *= x
end; @test f3b(ULogarithmic(1.0), ULogarithmic(3.0))[1] === ULogarithmic(3.0)

# ╔═╡ 8f169235-3bd1-4cc4-a083-79736d306ad5
example("computing x ^ 3 with logarithmic numbers")

# ╔═╡ dfc9d305-5bce-4555-bfa3-d8d61fe4ca09
@i function power3(y::ULogarithmic{T}, x::T) where T
	for i=1:3
		y *= convert(x)
	end
end; @test power3(ULogarithmic(1.0), 3.0)[1] ≈ 27

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

Functions ``f_{+-} ∈ \rm \{identity, +, -, *, /, ^\wedge, abs, abs2, sqrt, exp, log, sin, sinh, asin, cos, cosh,`` ``\rm acos, tan, tanh, atan, sincos, convert\}`` and ``f_{*/}∈\rm \{identity, +, -, *, /, ^\wedge, convert\}.`` 
Functions `FLIP`, `NEG`, `INV`, `HADAMARD`, `SWAP` and `y ⊻= f_{⊻}(args...)` are self-reversible (or reflexive). {`ROT`, `IROT`} and {`INC`, `DEC`}, {`y += f_{+-}(args...)`, `y -= f_{+-}(args...)`} and {`y *= f_{*/}(args...)`, `y /= f_{*/}(args...)`} are pair-wise reversible.

For Jacobians and Hessians defined on these instructions, please check this [blog post](https://giggleliu.github.io/2020/01/18/jacobians.html).
"""
# They are translated to `y += f(args...)` is translated to `PlusEq(f)(y, args...)`, `y -= f(args...)` is translated to `MinusEq(f)(y, args...)`, `y *= f(args...)` is translated to `MulEq(f)(y, args...)`, `y /= f(args...)` is translated to `DivEq(f)(y, args...)` and `y ⊻= f(args...)` is translated to `XorEq(f)(y, args...)`.

# ╔═╡ f6049c78-7468-47ce-a4a5-84fab34d115a
title2("How to create a new elementary reversible function")

# ╔═╡ b0f73825-bbb1-448c-b491-bf634fdd398a
md"To define a pair of elementary functions that **reverse to each other**,
1. declare two functions `f` and `g` that each of them defines a mapping ``\mathbb{R}^n \rightarrow \mathbb{R}^n``
2. use `@dual f g` to tell NiLang they are reversible to each other.
3. if you want to make `f` and `g` differentiable, you can specify backward rules on these two function by defining two mappings on ``\mathbb{G}^n\rightarrow \mathbb{G}^n``, where ``\mathbb{G}`` is a 2-tuple of ``\mathbb{R}`` (or `NiLang.AD.GVar`) in NiLang. It is similar to `ForwardDiff.Dual` (check [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl)) but defined for the backward pass.

To define a **self-reversible** elementary function
1. declare a functions `f` that defines a mapping ``\mathbb{R}^n \rightarrow \mathbb{R}^n``
2. use `@selfdual f` 
3. define the backward rule on `f` to make it differentiable.
"

# ╔═╡ 648cdcd6-f4f5-461f-a525-4b350cae9eb0
example("defining a new elementary function")

# ╔═╡ d6b1abd6-749d-4591-99e8-64aaa9199ab5
md"""
One can use the invertibility checker to check if the function is really reversible (under a certain tolerance `NiLangCore.GLOBAL_ATOL[]` = $(NiLangCore.GLOBAL_ATOL[])).
"""

# ╔═╡ f502b8c1-9b80-4e67-80e8-a64ddb88fb0f
@test NiLang.check_inv(M.new_forward, (3.0,))

# ╔═╡ 0bce342e-9a8e-4005-8b88-82da2d2c7163
md"""
To check of the gradients are properly defined, one can use `NiLang.AD.check_grad`
"""

# ╔═╡ fd9cf757-2698-4886-9f0a-c6c23ff0d331
@test NiLang.AD.check_grad(M.new_forward, (3.0,); iloss=1)

# ╔═╡ dda6652a-d063-4511-8041-e869bb88ca26
@test NiLang.AD.check_grad(M.new_backward, (3.0,); iloss=1)

# ╔═╡ f6bfa015-c101-45e8-995c-2bb6a3b7dc7d
title1("Types and arrays")

# ╔═╡ 8651d7ec-6bcd-4dbe-a062-c4bde32e5e91
md"
A Julia type can be accessed in NiLang if its default constructor is not overloaded, because NiLang requires the default constructor to \"modify\" a field of a immutable type. For example, one can modify the real and imaginary part of a complex number directly in NiLang."

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

# ╔═╡ edaa9fdb-3af8-4554-a701-0e3bff2107a5
md"It is also possbile to extract the fields directly."

# ╔═╡ 7551a880-340e-4e3f-815b-188e73f7eb9a
@i function complex_add(y::Complex{T}, x::Complex{T}) where T
    ((a, b), (c, d))::∅ ↔ ((@fields x), (@fields y))
	a += c
	b += d
    ((a, b), (c, d)) ↔ ((@fields x), (@fields y))::∅
end

# ╔═╡ 0489e51b-781f-4441-bb7f-ff3bd2e848ad
@test complex_add(1+2im, 3+4im) == (1+2im, 4+6im)

# ╔═╡ 7b0d30d6-39ff-4f6e-b13c-0ddbfcb576e5
md"Type cast is also possible"

# ╔═╡ 042297d8-6ab3-4ae6-b6e7-3b1ab2d5553b
@i function add4(a, b, c, d)
	complex_add(Complex{}(a, b), Complex{}(c, d))  # do not omit `{}`
end

# ╔═╡ 57d65a36-bfa8-4dc2-8e11-d87fa1324122
@test add4(1, 2, 3, 4) == (1, 2, 4, 6)

# ╔═╡ c21d81c3-981f-4472-ad61-d1661bfe5c4e
example("Implementing \"axpy\" function")

# ╔═╡ 99d6fe7b-d704-48f3-b115-2b3159a78068
md"One can modify the `Array` directly"

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

# ╔═╡ 0b37e505-12b7-45a9-a188-8a57cff055ec
md"One can use the swap between object fields and tuple."

# ╔═╡ c7f2f786-d9ff-43b4-baef-5a48d611aa1e
title1("Functions")

# ╔═╡ d6519029-231d-4c63-b47d-684462dab287
md"""
Macros on functions are partly supported, NiLang will render the body of the function first and pass it to the next macro. One can define an inline function like
```julia
@i @inline function f(args...)
	...
end
```

However, the generated functions are not yet supported.
"""

# ╔═╡ b3147f49-c0f8-40cc-a794-282f6950b392
md"
Functions can not be chained.
"

# ╔═╡ 3e1a9006-3930-44f6-9eed-584d78937a57
@test_throws LoadError macroexpand(NiLang, :(@i function f6a(x)
	INV(DEC(x))
end))

# ╔═╡ b5386441-617a-4cc9-a3e8-5d4067e9554e
md"For single input single out function, one can use `-`, `'` and the pipeline `var |> reversible function or data field getter` to make the function shorter."

# ╔═╡ bc19ff2a-d3ba-4a4b-959a-fe381015f2ef
@i function f6b(x)
	INV(-x |> SubConst(4.0))
end

# ╔═╡ 47c95524-cef5-4654-b9ed-324472707ef0
f6b(3.0)

# ╔═╡ 03b31baa-9925-4fcb-a95c-7e48bd7708ca
md"Why the result is not `1/(-3-4)`? This is because the `INV` operation operates on the *dataview* `-x |> SubConst(4.0)`,  rather than on `x` directly. Which mean when it computes the result `-1/7`, it tried to assign this value back to `-x |> SubConst(4.0)`. So it adds 4 to the result and negate it to make the change propagate to the original variable."

# ╔═╡ aacf63a2-9708-40db-8928-049621a7bbc4
md"## Control flows"

# ╔═╡ ad0097e7-c8ad-457a-82a9-18b998a9e9fb
md"""
#### If statement

The condition expression in `if` statement contains two parts, one is precondition and another is postcondition.

$(
twocol(md"
```julia
if (precondition1[, postcondition1])
	...
end
```
", md"
```julia
if (postcondition1[, precondition1])
	~(...)
end
```
")
)

where `...` are statements and `~(...)` are the backward execution of them.
"""

# ╔═╡ 94cd1345-3132-4882-86fe-d2429f610d1d
md"""If no postcondition is provided, it means the precondition is the same as the postcondition. It is translated to `if (cond, cond) ... end`. `elseif` is also supported, to avoid potential complications, we do not discuss it here."""

# ╔═╡ 4a558bd3-6e42-4c61-bd23-888b7f33ae25
@i function f7a(x)
	if x > 1
		x -= 1
	end
end; @test_throws InvertibilityError f7a(1.2)

# ╔═╡ 004f727a-e0c8-49cb-8858-dfdf4d3ac57a
@i function f7b(x, branch_keeper)
	branch_keeper ⊻= x > 1
	if (x > 1, branch_keeper)
		x += 1
	end
end; @test f7b(1.2, false)[1] == 2.2

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

# ╔═╡ 2f2b24ea-66d0-4b3b-a460-53b6b3f28ef0
md"The iterator length should not be changed during the iteration."

# ╔═╡ e9dbb64a-27b9-443a-b917-69d55c290235
@i function f7c(x, y)
	for i=1:length(x)
		POP!(x, y[i])
	end
end; @test_throws InvertibilityError f7c([1,2,3], [0,0,0])

# ╔═╡ 0d56ce96-81a5-4102-acbc-7d88f80adcb3
md"There is an `InvertibilityError` because the length of the `x` has been changed. The inverse execution will give incorrect result."

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
@i function f7d(x, i)
	@from i==0 while i<10
		i += 1
		x += 1
	end
end

# ╔═╡ 2ba68a0f-6e36-4ea2-a91d-6af43741bad1
f7d(1, 0)

# ╔═╡ 75cebaf1-38de-475f-892e-346fd2b46f6f
@test (~f7d)(11, 10) == (1, 0)

# ╔═╡ 72dcf2fe-eb48-4dee-8121-efafc87637e3
title2("Compute-copy-uncompute statement")

# ╔═╡ 84321198-93d4-4d22-8c0f-a5a10b884e1f
md"The *compute-copy-uncompute* statement is a widely used design pattern in reversible programming. 
We compute the forward pass for the result, then we copy the result to the output variable, and run the backward pass to erase intermediate results.



For example, to compute `y = x * exp(k)`, we might write the following code"

# ╔═╡ 32244789-afbf-4215-97cd-15483f438eee
@i function f7e(y, x, k)
	expk ← zero(k)
	expk += exp(k)
	y += x * expk
	# uncompute the ancilla and deallocate it
	expk -= exp(k)
	expk → zero(k)
end; @test f7e(0.0, 2.0, 3.0)[1] ≈ 2.0 * exp(3.0)

# ╔═╡ 4b7f0baf-0316-4da7-9ded-50c064ddbaa3
md"It is equivalent to the following statement that generates the backward pass automatically for you."

# ╔═╡ 0e02952c-7589-4606-b006-16a9f3e52ae1
@i function f7f(y, x, k)
	# record the forward pass
	@routine begin
		expk ← zero(k)
		expk += exp(k)
	end
	y += x * expk
	# reverse execute the recorded the program
	~@routine
end; @test f7f(0.0, 2.0, 3.0)[1] ≈ 2.0 * exp(3.0)

# ╔═╡ f0904d3f-1bf1-459c-9959-b53c0f774e3f
example("Computing Fibonacci numbers")

# ╔═╡ 19bb2af5-2a67-453d-82b0-7d3059b1fa47
md"The sequence of Fibonacci numbers are: 1, 1, 2, 3, 5, 8"

# ╔═╡ 5b5858bf-63ac-4e31-a516-055a9cd18ffe
@i function rfib(out!, n::T) where T
    @routine begin
		n1 ← zero(T)
		n2 ← zero(T)
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

# ╔═╡ 91f8cfc6-e261-4945-8506-eed8caa607c2
title1("Multi-threading and CUDA")

# ╔═╡ c82b3b5c-c4e2-4bf6-b4ec-0d05ba9a669b
@i function f8a(y::Vector, x::Vector)
	# check the size of `x` and `y`. `@assert` is not a valid statement in NiLang, so one should decorate it with `@safe` to tell the compiler, doing this is safe, do not check this statement.
	@safe @assert length(x) == length(y)
	@threads for i=1:length(y)
		y[i] += exp(x[i])
	end
end; @test f8a(zeros(3), [1.0, 2.0, 3.0])[1] ≈ [exp(1.0), exp(2.0), exp(3.0)]

# ╔═╡ 8c93a773-edc0-4ec2-88ef-1b58b7deddc5
title2("Shared read-write in parallel computing and autodiff")

# ╔═╡ 16d08950-0575-4a4b-afc8-11ddca3198c7
md"Let's take a look at a correct prallel code that compute `exp(x)` and broadcast it to the output."

# ╔═╡ 7c594d19-59fc-433a-bffa-c63bad46869e
@i function f8c(y::Vector, x::Real, z::Vector)
	@safe @assert length(z) == length(y)
	@threads for i=1:length(y)
		y[i] += x * z[i]
	end
end; @test f8c(zeros(3), 2.0, [1.0, 2.0, 3.0])[1] ≈ [2.0, 4.0, 6.0]

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

# ╔═╡ f80353d6-0dfe-4b0a-a1af-655d344473bf
title1("Resources")

# ╔═╡ 7ce31932-0447-4445-99aa-7ebced7d0bad
TableOfContents()

# ╔═╡ Cell order:
# ╟─2061b434-0ad1-46eb-a0c7-1a5f432bfa62
# ╟─a4e76427-f051-4b29-915a-fdfce3a299bb
# ╟─c2c7b4d4-f8c9-4ebf-8da2-0103f03136e7
# ╟─611b577f-4722-42bf-8f8e-aeb2fb30be71
# ╟─605872cf-f3fd-462e-a2b1-7d1c5ae45efd
# ╟─fb3dee44-5fa9-4773-8b7f-a83c44358545
# ╠═d941d6c2-55bf-11eb-0002-35c7474e4050
# ╠═70088425-6779-4a2d-ba6d-b0a34c8e93a6
# ╟─f0e94247-f615-472b-8218-3fa287b38aa1
# ╟─2581aa33-1dc5-40b1-aa9f-6a11cc750c93
# ╠═af738f89-3214-429c-9c7d-18a6ea0d9401
# ╟─4d98d529-1e05-49be-9209-f0d9fcc206f7
# ╠═48d7ebc1-5def-4a57-9ec1-3fc370a4543f
# ╟─e839547b-f47e-4a4e-b4d9-2c22921d80e4
# ╟─60575978-081a-4bca-a3ed-2b51cd6abc92
# ╠═f98305cb-4ba2-404a-a5c3-65510e059504
# ╟─e8cd6667-597f-458b-8465-1822e09a7891
# ╟─20145d75-004a-4c2f-b7ff-c400ca846d42
# ╟─b6dcd18c-606f-4340-b2ec-163e8bad03f5
# ╟─a1a29f34-f8a9-4e9f-9afe-7d0096771440
# ╟─90bd6ad4-3dd8-4e7c-b445-aed1d248a2ec
# ╠═c0259e48-1973-486c-a828-1fcd3e4331c6
# ╟─7bcc1bc6-5b2c-4aae-b71d-52482ae130e9
# ╠═fd5d65e7-3d83-45dd-87d7-ff1e1e769430
# ╟─8bbffa31-04a6-49ca-b36f-4d4140d75992
# ╠═a6f18c34-80ee-4b52-9ff8-f3c1b1d80f90
# ╠═a694132b-4f52-467f-8bc4-dc32fe2812db
# ╟─8c2c82f2-1240-4f2f-830e-ee8021c1a41a
# ╟─6203cf10-f8cc-4fb9-b814-7552b68c01dc
# ╠═f97a6bab-b9f9-4b95-98a9-381c51397526
# ╠═4ca48a2e-43da-457a-8e9f-6476097e4d7b
# ╟─2a2970f4-ab01-486b-89a2-6ff96f734018
# ╠═0b80d9be-53d7-4bf3-a558-659607af4709
# ╠═92362fda-bae2-4e35-bfe4-dcaea853d50b
# ╟─db9e7940-39f1-4ccf-ac70-146a521daa6e
# ╟─93936612-1447-4114-b864-aba43adef4bd
# ╠═b2add70c-c5d6-4e0f-a153-43e21a197181
# ╠═5c03d5a5-99f0-4efd-9a32-ce6d7c2b266c
# ╟─2349e3ea-3053-42a4-b9d9-f97a76e4abd7
# ╟─269f18ee-3cd8-466a-a522-7c624503e31b
# ╠═89139719-c478-4066-9452-f9893f36d561
# ╠═d620c5ee-7d9c-4d3f-9e87-0c828dfab9ca
# ╟─255d01b9-a873-4e63-9298-9d8f073348b0
# ╟─f6cf1729-766c-4ed7-b004-c8c8ec6c7e07
# ╠═3645d672-423f-4ac8-805f-0452793fee5a
# ╠═c2a0024e-11dd-4ef7-8346-4374d98cafc0
# ╟─b20004e9-3c73-4dfb-8fd5-f377786fd53b
# ╠═5c1952b1-5016-4c87-b23c-8e6a235bf8cd
# ╠═8e4470ee-01da-4547-b091-c4f65cd729b0
# ╟─4601df35-679f-465d-9191-c18748b2fd83
# ╟─4fc72b9d-19a2-40f1-a4a8-5e97d3d5e529
# ╠═a0fde16f-8454-4f5c-a29c-a9e415c0c311
# ╟─633ff8f3-8d93-4f73-bec2-c42070e6ece9
# ╠═57f2d890-b5a0-47b7-9e3d-af4d03b10605
# ╟─34063cd0-171e-46ce-80dd-52a341fa50a1
# ╠═5d5d01db-8ff9-434c-8771-1fec6393e1fb
# ╠═10d85a50-f2f9-403e-8f6c-baef61cf702a
# ╟─b52648bf-a28a-48af-8912-31729d943ce0
# ╟─f45db10f-a836-40f3-9d8d-054ea6540e87
# ╟─1903563e-ccc2-44d9-9dbe-e5dede275b3c
# ╟─7ec577d4-1e19-484e-ac1f-5da3f47d1ec4
# ╟─583f2585-15a3-47c6-a70e-e2f002754028
# ╠═60e6ff80-3593-4ae4-a273-914847f692db
# ╠═9e5cfd68-b58d-4d83-aae2-447e5f805c97
# ╠═dc85a942-cf52-4405-ad03-32a768e1b6e7
# ╟─5cdd346b-10a5-485c-ba78-4c0b3cb0e02f
# ╠═af9287b7-6131-46f6-beb8-6885e55e1975
# ╠═e20eeabf-1c80-431e-8cfc-4d1b79c52b5a
# ╟─e45d53b9-6f02-4293-ab4f-85bd751993ad
# ╟─90d30eea-53de-48a0-9700-ff35681fdf38
# ╠═390f58a5-6f5f-4d3a-bb16-ba04e43a07e7
# ╠═2b57443e-a516-434b-be86-80616a98e2f5
# ╟─fc2e27f9-b7ba-44cc-a953-6745548ad733
# ╠═fc744931-360b-4478-9f77-c50f048de243
# ╠═9a152b36-f377-44da-9700-ca9e05e365ff
# ╟─0863bd06-cc70-4dde-b3b2-0a466805a356
# ╟─a0bae195-04e1-4642-9e14-fe4691e0906b
# ╠═20d6e8a0-2cf5-48ad-9549-60506b42b970
# ╠═7dee5748-ed73-4e13-aa80-7a50efbc8449
# ╟─4c719e9b-641e-404e-9ab7-59e89135f3ba
# ╠═77947e00-42c3-4c9e-b62a-b4b29489db43
# ╠═2614127d-34fb-4c3d-b678-42693f3c9341
# ╟─8f169235-3bd1-4cc4-a083-79736d306ad5
# ╠═dfc9d305-5bce-4555-bfa3-d8d61fe4ca09
# ╟─c4cd9f88-9cd6-4364-b016-78f90aba6a66
# ╟─f6049c78-7468-47ce-a4a5-84fab34d115a
# ╟─b0f73825-bbb1-448c-b491-bf634fdd398a
# ╟─648cdcd6-f4f5-461f-a525-4b350cae9eb0
# ╠═8c2c4fa6-172f-4dde-a279-5d0aecfdbe46
# ╟─d6b1abd6-749d-4591-99e8-64aaa9199ab5
# ╠═f502b8c1-9b80-4e67-80e8-a64ddb88fb0f
# ╟─0bce342e-9a8e-4005-8b88-82da2d2c7163
# ╠═fd9cf757-2698-4886-9f0a-c6c23ff0d331
# ╠═dda6652a-d063-4511-8041-e869bb88ca26
# ╟─f6bfa015-c101-45e8-995c-2bb6a3b7dc7d
# ╟─8651d7ec-6bcd-4dbe-a062-c4bde32e5e91
# ╟─9f5f9de3-9558-4c18-9d98-b77d19b570ec
# ╠═6dfcfa19-f78f-4dac-89f7-d3c5dbe17987
# ╟─edaa9fdb-3af8-4554-a701-0e3bff2107a5
# ╠═7551a880-340e-4e3f-815b-188e73f7eb9a
# ╠═0489e51b-781f-4441-bb7f-ff3bd2e848ad
# ╟─7b0d30d6-39ff-4f6e-b13c-0ddbfcb576e5
# ╠═042297d8-6ab3-4ae6-b6e7-3b1ab2d5553b
# ╠═57d65a36-bfa8-4dc2-8e11-d87fa1324122
# ╟─c21d81c3-981f-4472-ad61-d1661bfe5c4e
# ╟─99d6fe7b-d704-48f3-b115-2b3159a78068
# ╠═1950ff70-54eb-4ece-a26d-a23fd0e90f5a
# ╟─21458f81-9007-46f8-92e0-7a17c60beb36
# ╠═7813f4ce-6e98-45f3-94a8-7f5981129f2b
# ╟─59ec7cb7-6011-456d-9f57-a55bb8ea51a0
# ╟─0b37e505-12b7-45a9-a188-8a57cff055ec
# ╟─c7f2f786-d9ff-43b4-baef-5a48d611aa1e
# ╟─d6519029-231d-4c63-b47d-684462dab287
# ╟─b3147f49-c0f8-40cc-a794-282f6950b392
# ╠═3e1a9006-3930-44f6-9eed-584d78937a57
# ╟─b5386441-617a-4cc9-a3e8-5d4067e9554e
# ╠═bc19ff2a-d3ba-4a4b-959a-fe381015f2ef
# ╠═47c95524-cef5-4654-b9ed-324472707ef0
# ╟─03b31baa-9925-4fcb-a95c-7e48bd7708ca
# ╟─aacf63a2-9708-40db-8928-049621a7bbc4
# ╟─ad0097e7-c8ad-457a-82a9-18b998a9e9fb
# ╟─94cd1345-3132-4882-86fe-d2429f610d1d
# ╠═4a558bd3-6e42-4c61-bd23-888b7f33ae25
# ╠═004f727a-e0c8-49cb-8858-dfdf4d3ac57a
# ╟─4c03cde9-b643-40ff-b275-f1795f88949e
# ╟─fae7c74e-d25e-4c1e-ac97-199e6dae3365
# ╟─2f2b24ea-66d0-4b3b-a460-53b6b3f28ef0
# ╠═e9dbb64a-27b9-443a-b917-69d55c290235
# ╟─0d56ce96-81a5-4102-acbc-7d88f80adcb3
# ╟─95c41bd1-e50a-42e8-93c3-3b754a458c13
# ╟─b8629aeb-6c9a-44ed-87a1-9ab22d9485ed
# ╟─3b211406-041f-4b41-acae-3958e4a37224
# ╠═62522772-cb59-4d13-acdd-d5067b223910
# ╠═2ba68a0f-6e36-4ea2-a91d-6af43741bad1
# ╠═75cebaf1-38de-475f-892e-346fd2b46f6f
# ╟─72dcf2fe-eb48-4dee-8121-efafc87637e3
# ╟─84321198-93d4-4d22-8c0f-a5a10b884e1f
# ╠═32244789-afbf-4215-97cd-15483f438eee
# ╟─4b7f0baf-0316-4da7-9ded-50c064ddbaa3
# ╠═0e02952c-7589-4606-b006-16a9f3e52ae1
# ╟─f0904d3f-1bf1-459c-9959-b53c0f774e3f
# ╟─19bb2af5-2a67-453d-82b0-7d3059b1fa47
# ╠═5b5858bf-63ac-4e31-a516-055a9cd18ffe
# ╠═95060588-f24b-4eeb-9b0b-ed7159962a3c
# ╟─91f8cfc6-e261-4945-8506-eed8caa607c2
# ╠═0e1ba158-a6bc-401c-9ba7-ed78020ad068
# ╠═c82b3b5c-c4e2-4bf6-b4ec-0d05ba9a669b
# ╟─8c93a773-edc0-4ec2-88ef-1b58b7deddc5
# ╟─16d08950-0575-4a4b-afc8-11ddca3198c7
# ╠═7c594d19-59fc-433a-bffa-c63bad46869e
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
# ╟─f80353d6-0dfe-4b0a-a1af-655d344473bf
# ╟─7ce31932-0447-4445-99aa-7ebced7d0bad

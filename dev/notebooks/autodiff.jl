### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ f11023e5-8f7b-4f40-86d3-3407b61863d9
begin
	using PlutoUI, Viznet, Compose, Plots
	function shrink(a, b, da, db)
		d = b .- a
		r = sqrt(sum(abs2, d))
		unitd = d ./ r
		a .+ unitd .* da, b .- unitd .* db
	end
end;

# ╔═╡ ce44f8bd-692e-4eab-9ba4-055b25e40c81
using ForwardDiff: Dual

# ╔═╡ 9a46597c-b1ee-4e3b-aed1-fd2874b6e77a
using BenchmarkTools

# ╔═╡ ccd38f52-104d-434a-aea3-dd94e571374f
using NiLang

# ╔═╡ f4230251-ba54-434a-b86b-f972c7389217
using MacroTools

# ╔═╡ 69dc2685-b70f-4a81-af30-f02e0054bd52
using NiLang.AD

# ╔═╡ 200f1848-0980-4185-919a-93ab2e7f788f
using SparseArrays

# ╔═╡ 30c191c5-642b-4062-98f3-643d314a054d
using LinearAlgebra

# ╔═╡ 864dbde7-b689-4165-a08e-6bbbd72190de
using Test

# ╔═╡ a1ef579e-4b66-4042-944e-7e27c660095e
md"""
```math
\newcommand{\comment}[1]{{\bf  \color{blue}{\text{◂~ #1}}}}
```
"""

# ╔═╡ 100b4293-fd1e-4b9c-a831-5b79bc2a5ebe
begin
	# left right layout
	function leftright(a, b; width=600)
		HTML("""
<style>
table.nohover tr:hover td {
   background-color: white !important;
}</style>
			
<table width=$(width)px class="nohover" style="border:none">
<tr>
	<td>$(html(a))</td>
	<td>$(html(b))</td>
</tr></table>
""")
	end
	
	# up down layout
	function updown(a, b; width=nothing)
		HTML("""<table class="nohover" style="border:none" $(width === nothing ? "" : "width=$(width)px")>
<tr>
	<td>$(html(a))</td>
</tr>
<tr>
	<td>$(html(b))</td>
</tr></table>
""")
	end
	
	function highlight(str)
		HTML("""<span style="background-color:yellow">$(str)</span>""")
	end
end;

# ╔═╡ 9d11e058-a7d0-11eb-1d78-6592ff7a1b43
md"# An introduction to automatic differentiation

-- GiggleLiu"

# ╔═╡ b73157bf-1a77-47b8-8a06-8d6ec2045023
html"<button onclick='present()'>present</button>"

# ╔═╡ ec13e0a9-64ff-4f66-a5a6-5fef53428fa1
md"""
* What is automatic differentiation (AD)?
    * A true history of AD
    * Forward mode AD
    * Reverse mode AD 
        * primitves on tensors (including Jax, pytorch et al.)
        * primitves on elementary instructions (usually source code transformation based)
        * defined on a reversible program
* Some applications in **scientific computing**
    * solving the graph embedding problem
    * inverse engineering a hamiltonian
    * obtaining maximum independent set (MIS) configurations
    * towards differentiating `expmv` ``\comment{will be used in our emulator}``
"""

# ╔═╡ f8b0d1ce-99f7-4729-b46e-126da540cbbe
md"""
## The true history of automatic differentiation
"""

# ╔═╡ 435ac19e-1c0c-4ee5-942d-f2a97c8c4d80
md"""
* 1964 ~ Robert Edwin Wengert, A simple automatic derivative evaluation program. ``\comment{first forward mode AD}``
* 1970 ~ Seppo Linnainmaa, Taylor expansion of the accumulated rounding error. ``\comment{first backward mode AD}``
* 1986 ~ Rumelhart, D. E., Hinton, G. E., and Williams, R. J., Learning representations by back-propagating errors.
* 1992 ~ Andreas Griewank, Achieving logarithmic growth of temporal and spatial complexity in reverse automatic differentiation. ``\comment{foundation of source code transformation based AD.}``
* 2000s ~ The boom of tensor based AD frameworks for machine learning.
* 2018 ~ People re-invented AD as differential programming ([wiki](https://en.wikipedia.org/wiki/Differentiable_programming) and this [quora answer](https://www.quora.com/What-is-Differentiable-Programming).)
![](https://qph.fs.quoracdn.net/main-qimg-fb2f8470f2120eb49c8142b08d9c4132)
* 2020 ~ Me, Differentiate everything with a reversible embeded domain-specific language ``\comment{AD based on reversible programming}``.
"""

# ╔═╡ 48ecd619-d01d-43ff-8b52-7c2566c3fa2b
md"## Forward mode automatic differentiation"

# ╔═╡ 4878ce45-40ff-4fae-98e7-1be41e930e4d
md"""
Forward mode AD attaches a infitesimal number $\epsilon$ to a variable, when applying a function $f$, it does the following transformation
```math
\begin{align}
    f(x+g \epsilon) = f(x) + f'(x) g\epsilon + \mathcal{O}(\epsilon^2)
\end{align}
```

The higher order infinitesimal is ignored. 

**In the program**, we can define a *dual number* with two fields, just like a complex number
```
f((x, g)) = (f(x), f'(x)*g)
```
"""

# ╔═╡ b2c1936c-2c27-4fbb-8183-e38c5e858483
res = sin(Dual(π/4, 2.0))

# ╔═╡ 8be1b812-fcac-404f-98aa-0571cb990f34
res === Dual(sin(π/4), cos(π/4)*2.0)

# ╔═╡ 33e0c762-c75e-44aa-bfe2-bff92dd1ace8
md"
We can apply this transformation consecutively, it reflects the chain rule.
```math
\begin{align}
\frac{\partial \vec y_{i+1}}{\partial x} &= \boxed{\frac{\partial \vec y_{i+1}}{\partial \vec y_i}}\frac{\partial \vec y_i}{\partial x}\\
&\text{local Jacobian}
\end{align}
```
"

# ╔═╡ c59c35ee-1907-4736-9893-e22c052150ca
let
	lb = textstyle(:math, fontsize(8), width=0.5, height=0.5)
	tb = textstyle(:default, fontsize(10), Compose.font("monospace"))
	tb_big = textstyle(:default, fontsize(3.5), fill("white"), Compose.font("monospace"))
	nb = nodestyle(:circle, fill("white"), Compose.stroke("black"); r=0.08)
	tri = nodestyle(:triangle, Compose.stroke("transparent"), fill("black"); r=0.02)
	eb = bondstyle(:default, linewidth(0.5mm))
	ebr = bondstyle(:default, Compose.stroke("red"), linewidth(0.5mm))
	ebd = bondstyle(:default, linewidth(0.5mm), dashed=true)
	eba = bondstyle(:default, linewidth(0.5mm), Compose.arrow(), Compose.stroke("red"), Compose.fill("red"))
		
	function arrow(x, y)
		mid = (x .+ y) ./ 2
		t = nodestyle(:triangle, fill("red"), θ=π/2-atan((y .- x)...)-1π/6)
		ebr >> (x, y)
		t >> mid
	end
	
	Compose.set_default_graphic_size(15cm, 5cm)
	x = (0.1, 0.5)
	fi0 = (0.35, 0.5)
	fi1 = (0.7, 0.5)
	fi2 = (1.0, 0.5)
	img = canvas() do
		nb >> fi0
		nb >> fi1
		lb >> (fi0 .- (0.05, 0.1), "f_{i-1}")
		lb >> (fi1 .- (0.02, 0.1), "f_{i}")
		lb >> (x, "x")
		lb >> ((fi1 .+ fi0) ./ 2 .- (0.02, 0.0), raw"\vec{y}_{i}")
		lb >> ((fi1 .+ fi2) ./ 2 .- (0.05, 0.0), raw"\vec{y}_{i+1}")
		lb >> ((fi1 .+ fi2) ./ 2 .- (0.05, 0.0), "\\vec{y}_{i+1}")
		lb >> (x .- (0.00, 0.25), raw"\color{red}{1}")
		lb >> ((fi1 .+ fi0) ./ 2 .- (0.05, 0.45), raw"\color{red}{\frac{\partial \vec{y}_{i}}{\partial x}}")
		lb >> ((fi1 .+ fi2) ./ 2 .- (0.08, 0.45), raw"\color{red}{\frac{\partial \vec{y}_{i+1}}{\partial x}}")
		ebd >> (x, fi0)
		eb >> (fi0, fi1)
		eb >> (fi1, fi2)
		#arrow((fi1 .+ fi0) ./ 2 .+ (0.08, -0.3), (fi1 .+ fi2) ./ 2 .+ (-0.08, -0.3))
		arrow((fi1 .+ fi0) ./ 2 .+ (0.08, -0.3), (fi1 .+ fi2) ./ 2 .+ (-0.08, -0.3))
	end
	img
end

# ╔═╡ 0ae13734-b826-4dbf-93d1-11044ce88bd4
x_ = Dual(π/4, 1.0)

# ╔═╡ 99187515-c8be-49c2-8d70-9c2998d9993c
sin(x_)

# ╔═╡ 78ca6b08-84c4-4e4d-8412-ae6c28bfafce
md"when automatic comes in"

# ╔═╡ f12b25d8-7c78-4686-b46d-00b34e565605
let
	x = Dual(π/4, 1.0)
	z = Dual(1.1)
	for i=1:10
		x = sin(x) * z
	end
	x
end

# ╔═╡ d90c3cc9-084d-4cf7-9db7-42cea043030b
md"""
**Example:** Computing two gradients $\frac{\partial z\sin x}{\partial x}$ and $\frac{\partial \sin^2x}{\partial x}$ at one sweep
"""

# ╔═╡ 93c98cb2-18af-47df-afb3-8c5a34b4723c
let
	lb = textstyle(:math, fontsize(8), width=1.0, height=0.5)
	tb = textstyle(:default, fontsize(3.5), Compose.font("monospace"))
	tb_big = textstyle(:default, fontsize(4.5), fill("white"), Compose.font("monospace"))
	nb = nodestyle(:circle, fill("black"), Compose.stroke("transparent"); r=0.05)
	tri = nodestyle(:triangle, Compose.stroke("transparent"), fill("black"); r=0.02)
	eb = bondstyle(:default, linewidth(0.5mm))
	
	x_x = (0.1, 0.25)
	x_y = (0.9, 0.5)
	x_y2 = (0.9, 0.25)
	x_z = (0.3, 0.5)
	x_sin = (0.3, 0.25)
	x_mul = (0.5, 0.5)
	x_square = (0.5, 0.25)
	
	function arrow(x, y)
		mid = (x .+ y) ./ 2
		t = nodestyle(:triangle, θ=π/2-atan((y .- x)...)-1π/6)
		eb >> (x, y)
		t >> mid
	end

	img = canvas() do
		nb >> x_sin
		nb >> x_mul
		nb >> x_square
		tb_big >> (x_sin, "sin")
		tb_big >> (x_mul .+ (0, 0.01), "*")
		tb_big >> (x_square, "^2")
		arrow(x_sin, x_mul)
		arrow(x_x, x_sin)
		arrow(x_mul, x_y)
		arrow(x_square, x_y2)
		arrow(x_z, x_mul)
		arrow(x_sin, x_square)
		tb >> ((x_x .+ x_sin) ./ 2 .- (0.02, 0.04), "x+ϵˣ")
		tb >> ((x_sin .+ x_mul) ./ 2 .- (0.08, 0.04), "sin(x)+cos(x)*ϵˣ")
		tb >> ((x_y .+ x_mul) ./ 2 .- (-0.04, 0.055), "z*sin(x)\n+z*cos(x)*ϵˣ")
		tb >> ((x_y2 .+ x_square) ./ 2 .- (-0.04, 0.055), "sin(x)^2\n+2*sin(x)*cos(x)*ϵˣ")
		tb >> ((x_z .+ x_mul) ./ 2 .- (0.05, 0.02), "z")
	end
	
	Compose.set_default_graphic_size(100mm, 100mm/2)
	Compose.compose(context(0, -0.15, 1, 2), img)
end

# ╔═╡ 2dc74e15-e2ea-4961-b43f-0ada1a73d80a
md"so the gradients are $z\cos x$ and $2\sin x\cos x$"

# ╔═╡ 7ee75a15-eaea-462a-92b6-293813d2d4d7
md"""
**What if we want to compute gradients for multiple inputs?**

The computing time grows **linearly** as the number of variables that we want to differentiate. But does not grow significantly with the number of outputs.
"""

# ╔═╡ 02a25b73-7353-43b1-8738-e7ca472d0cc7
md"""
## Reverse mode automatic differentiation

"""

# ╔═╡ 2afb984f-624e-4381-903f-ccc1d8a66a17
md"On the other side, the back-propagation can differentiate **many inputs** with respect to a **single output** efficiently"

# ╔═╡ 7e5d5e69-90f2-4106-8edf-223c150a8168
md"""
```math
\begin{align}
    \frac{\partial \mathcal{L}}{\partial \vec y_i} = \frac{\partial \mathcal{L}}{\partial \vec y_{i+1}}&\boxed{\frac{\partial \vec y_{i+1}}{\partial \vec y_i}}\\
&\text{local jacobian?}
\end{align}
```
"""

# ╔═╡ 92d7a938-9463-4eee-8839-0b8c5f762c79
let
	lb = textstyle(:math, fontsize(8), width=0.5, height=0.5)
	tb = textstyle(:default, fontsize(10), Compose.font("monospace"))
	tb_big = textstyle(:default, fontsize(3.5), fill("white"), Compose.font("monospace"))
	nb = nodestyle(:circle, fill("white"), Compose.stroke("black"); r=0.08)
	tri = nodestyle(:triangle, Compose.stroke("transparent"), fill("black"); r=0.02)
	eb = bondstyle(:default, linewidth(0.5mm))
	ebr = bondstyle(:default, Compose.stroke("red"), linewidth(0.5mm))
	ebd = bondstyle(:default, linewidth(0.5mm), dashed=true)
	eba = bondstyle(:default, linewidth(0.5mm), Compose.arrow(), Compose.stroke("red"), Compose.fill("red"))
		
	function arrow(x, y)
		mid = (x .+ y) ./ 2
		t = nodestyle(:triangle, fill("red"), θ=π/2-atan((y .- x)...)-1π/6)
		ebr >> (x, y)
		t >> mid
	end
	
	Compose.set_default_graphic_size(15cm, 5cm)
	x = (0.1, 0.5)
	fi0 = (0.35, 0.5)
	fi1 = (0.7, 0.5)
	fi2 = (0.9, 0.5)
	img = canvas() do
		nb >> fi0
		nb >> fi1
		lb >> (fi0 .- (0.02, 0.1), "f_{i}")
		lb >> (fi1 .- (0.05, 0.1), "f_{i+1}")
		lb >> (fi2 .- (0.05, 0.0), raw"\mathcal{L}")
		lb >> ((fi0 .+ x) ./ 2 .- (0.05, 0.0), raw"\vec{y}_{i}")
		lb >> ((fi0 .+ fi1) ./ 2 .- (0.05, 0.0), raw"\vec{y}_{i+1}")
		lb >> ((fi0 .+ fi1) ./ 2 .- (0.05, 0.0), "\\vec{y}_{i+1}")
		lb >> (fi2 .- (0.05, 0.25), raw"\color{red}{1}")
		lb >> ((fi0 .+ x) ./ 2 .- (0.08, 0.45), raw"\color{red}{\frac{\partial \mathcal{L}}{\partial \vec{y}_{i}}}")
		lb >> ((fi0 .+ fi1) ./ 2 .- (0.08, 0.45), raw"\color{red}{\frac{\partial \mathcal{L}}{\partial \vec{y}_{i+1}}}")
		ebd >> (fi1, fi2)
		eb >> (fi0, fi1)
		eb >> (x, fi0)
		#arrow((fi1 .+ fi0) ./ 2 .+ (0.08, -0.3), (fi1 .+ fi2) ./ 2 .+ (-0.08, -0.3))
		arrow( (fi0 .+ fi1) ./ 2 .+ (-0.08, -0.3), (fi0 .+ x) ./ 2 .+ (0.05, -0.3),)
	end
	img
end

# ╔═╡ 4b1a0b59-ddc6-4b2d-b5f5-d92084c31e46
md"### How to visit local Jacobians in the reversed order? "

# ╔═╡ 81f16b8b-2f0b-4ba3-8c26-6669eabf48aa
md"The naive approach is to store everything."

# ╔═╡ fb6c3a48-550a-4d2e-a00b-a1e40d86b535
md"""
**Example:** Computing the gradient $\frac{\partial z\sin x}{\partial x}$ and $\frac{\partial z\sin x}{\partial z}$ by back propagating cached local information.
"""

# ╔═╡ ab6fa4ac-29ed-4722-88ed-fa1caf2072f3
let
	lb = textstyle(:math, fontsize(10), width=1.0, height=0.5)
	tb = textstyle(:default, fontsize(3.5), Compose.font("monospace"))
	tbc = textstyle(:default, fontsize(3.5), fill("red"), Compose.font("monospace"))
	tb_big = textstyle(:default, fontsize(4), fill("white"), Compose.font("monospace"))
	nb = nodestyle(:circle, fill("black"), Compose.stroke("transparent"); r=0.05)
	tri = nodestyle(:triangle, Compose.stroke("transparent"), fill("black"); r=0.02)
	eb = bondstyle(:default, linewidth(0.5mm))
	
	x_x = (0.1, 0.2)
	x_y = (0.9, 0.5)
	x_z = (0.1, 0.7)
	x_sin = (0.3, 0.3)
	x_mul = (0.5, 0.5)

	function arrow(x, y)
		mid = (x .+ y) ./ 2
		t = nodestyle(:triangle, θ=π/2-atan((y .- x)...)-1π/6)
		eb >> (x, y)
		t >> mid
	end
	img1 = canvas() do
		nb >> x_sin
		nb >> x_mul
		tb_big >> (x_sin, "sin")
		tb_big >> (x_mul .+ (0, 0.01), "*")
		arrow(x_sin, x_mul)
		arrow(x_x, x_sin)
		arrow(x_mul, x_y)
		arrow(x_z, x_mul)
		tb >> ((x_x .+ x_sin) ./ 2 .- (0.0, 0.1), "x \n push(Σ,x)")
		tb >> ((x_sin .+ x_mul) ./ 2 .- (-0.15, 0.04), "s = sin(x) \n push(Σ,s)")
		tb >> ((x_y .+ x_mul) ./ 2 .- (-0.05, 0.04), "y = z*sin(x)")
		tb >> ((x_z .+ x_mul) ./ 2 .- (0.05, 0.07), "z\n push(Σ,z)")
	end
	img2 = canvas() do
		nb >> x_sin
		nb >> x_mul
		tb_big >> (x_sin, "sin")
		tb_big >> (x_mul .+ (0, 0.01), "*")
		arrow(x_mul, x_sin)
		arrow(x_sin, x_x)
		arrow(x_y, x_mul)
		arrow(x_mul, x_z)
		tb >> ((x_x .+ x_sin) ./ 2 .- (0.0, 0.1), "x = pop(Σ)\nx̄ = cos(x)*s̄")
		tb >> ((x_sin .+ x_mul) ./ 2 .- (-0.12, 0.04), "z = pop(Σ)\ns̄ = z*ȳ")
		tb >> ((x_y .+ x_mul) ./ 2 .- (-0.05, 0.06), "y\nȳ=1")
		tb >> ((x_z .+ x_mul) ./ 2 .- (0.05, 0.07), "s = pop(Σ)\nz̄ = s*ȳ")
	end
	
	Compose.set_default_graphic_size(150mm, 75mm/1.4)
	Compose.compose(context(), 
	(context(0, -0.1, 0.5, 1.4), img1),
	(context(0.5, -0.1, 0.5, 1.4), img2)
	)
end

# ╔═╡ 8e72d934-e307-4505-ac82-c06734415df6
md"Here, we use $\overline y$ for $\frac{\partial \mathcal{L}}{\partial y}$, which is also called the adjoint."

# ╔═╡ e6ff86a9-9f54-474b-8111-a59a25eda506
md"### Primitives on different scales"

# ╔═╡ 9c1d9607-a634-4350-aacd-2d40984d647d
md"We call the leaf nodes defining AD rules \"**primitives**\""

# ╔═╡ 63db2fa2-50b2-4940-b8ee-0dc6e3966a57
md"
**Design Decision**

* A: If we define primitives on **arrays**, we need tons of manually defined backward rules. (Jax, Pytorch, Zygote.jl, ReverseDiff.jl et al.)
* B: If we define primitives on **scalar instructions**, we will have worse tensor performance. (Tapenade, Adept, NiLang et al.)

*Note*: Here, implementing AD on scalars means specifically the **optimal checkpointing** approach, rather than a package like Jax, Zygote and ReverseDiff that having scalar support.
"

# ╔═╡ 693167e7-e80c-401d-af89-55b5fae30848
let
	w, h = 0.22, 0.1
	lb = Compose.compose(context(), polygon([(-w, -h), (-w, h), (w, h), (w, -h)]), Compose.stroke("transparent"))
	lb2 = Compose.compose(context(), polygon([(-w, -h), (-w, h), (w, h), (w, -h)]), Compose.stroke("transparent"), fill("red"))
	tb = Compose.compose(context(), Compose.text(0.0, 0.0, ""), fontsize(3), Compose.font("monospace"))
	tb_big = textstyle(:default, fontsize(3), fill("white"), Compose.font("monospace"))
	eb = bondstyle(:default, linewidth(0.5mm))
	ar = bondstyle(:default, linewidth(0.3mm), Compose.arrow())
	xprog = (0.25, 0.15)
	xtensors = (0.25, 0.5)
	t1 = (0.5, 0.15)
	t2 = (0.5, 0.5)
	t3 = (0.5, 0.85)
	xscalars2 = (0.25, 0.85)
	
	function box(loc, text; color="black")
		(color=="black" ? lb : lb2) >> loc
		tb_big >> (loc, text)
	end
	Compose.set_default_graphic_size(10cm, 5cm)
	canvas() do
		box(xprog, "Program")
		ar >> (xprog, xtensors .+ (0, -h-0.03))
		#ar >> (xprog, xscalars .+ (-w/2, -h-0.03))
		ar >> (xtensors, xscalars2 .+ (0, -h-0.05))
		box(xtensors, "Functions on arrays")
		#box(xscalars, "Functions on Scalars")
		box(xscalars2, "Finite instructions"; color="red")
		tb >> (t1, "Neural networks")
		tb >> (t2, "matrix multiplication")
		tb >> (t3, "+, -, *")
	end
end

# ╔═╡ 4cd70901-2142-4868-9a33-c46ca0d064ec
html"""
<table>
<tr>
<th width=200></th>
<th width=300>on tensors</th>
<th width=300>on finite instructions</th>
</tr>
<tr style="vertical-align:top">
<td>meaning</td>
<td>defining backward rules manully for functions on tensors</td>
<td>defining backward rules on a limited set of basic scalar operations, and generate gradient code using source code transformation</td>
</tr>
<tr style="vertical-align:top">
<td>pros and cons</td>
<td>
<ol>
<li style="color:green">Good tensor performance</li>
<li style="color:green">Mature machine learning ecosystem</li>
<li style="color:red">Need to define backward rules manually</li>
</ol>
</td>
<td>
<ol>
<li style="color:green">Reasonalbe scalar performance</li>
<li style="color:red">hard to utilize GPU kernels (except NiLang.jl) and BLAS</li>
</ol>
</td>
<td>
</td>
</tr>
<tr style="vertical-align:top">
<td>packages</td>
<td>Jax<br>PyTorch</td>
<td><a href="http://tapenade.inria.fr:8080/tapenade/">Tapenade</a><br>
<a href="http://www.met.reading.ac.uk/clouds/adept/">Adept</a><br>
<a href="https://github.com/GiggleLiu/NiLang.jl">NiLang.jl</a>
</td>
</tr>
</table>
"""

# ╔═╡ 89018a35-76f4-4f23-b15a-a600db046d6f
md"## A book"

# ╔═╡ 1d219222-0778-4c37-9182-ed5ccbb3ef32
leftright(html"""
<img src="https://images-na.ssl-images-amazon.com/images/I/51+dn97bfKL._SY344_BO1,204,203,200_.jpg"/>
""", md"**Evaluating derivatives: principles and techniques of algorithmic differentiation**
	
By: Griewank, Andreas, and Andrea Walther
(2008)")

# ╔═╡ 4ff09f7c-aeac-48bd-9d58-8446137c3acd
md"""
## The AD ecosystem in Julia

Please check JuliaDiff: [https://juliadiff.org/](https://juliadiff.org/)

A short list:
* Forward mode AD: ForwardDiff.jl
* Reverse mode AD (tensor): ReverseDiff.jl/Zygote.jl
* Reverse mode AD (scalar): NiLang.jl

Warnings
* The main authors of `Tracker`, `ReverseDiff` and `Zygote` are not maintaining them anymore.
"""
#=
|       |   Rules | Favors Tensor? | Type |
| ---- | ---- | --- | --- |
|  Zygote   |  C  |  ✓   |   R     |
|  ReverseDiff  |  D    | ✓    | R |
|  Nabla   |  D→C  |   ✓  |   R     |
|  Tracker  |  D    | ✓    | R |
|  Yota   |  C  |  ✓   |     R   |
|  NiLang   |  -  |  ×   |  R      |
|  Enzyme   |  -  |  ×   |  R      |
|  ForwardDiff   |  -  |  ×   |    F    |
|  Diffractor   |  ?  |  ?   |  ?      |

* R: reverse mode
* F: forward mode
* C: ChainRules
* D: DiffRules
"""
=#

# ╔═╡ ea44037b-9359-4fbd-990f-529d88d54351
md"# Quick summary
1. The history of AD is longer than many people have thought. People are most familar with *reverse mode AD with primitives implemented on tensors* that brings the boom of machine learning. There are also AD frameworks that can differentiate a general program directly, which does not require users defining AD rules manually.
2. **Forward mode AD** propagate gradients forward, it has a computational overhead propotional to the number of input parameters.
2. **Backward mode AD** propagate gradients backward, it has a computational overhead propotional to the number of output parameters.
    * primitives on **tensors** v.s. **scalars**
    * it is very expensive to reverse the program
4. Julia has one of the most active AD community!

#### Forward v.s. Backward
when is forward mode AD more useful?

* It is often combined with backward mode AD for obtaining Hessians (forward over backward).
* Having <20 input parameters.

when is backward mode AD more useful?
* In most variational optimizations, especially when we are training a neural network with ~ 100M parameters.
"

# ╔═╡ e731a8e3-6462-4a60-83e9-6ab7ddfff50e
md"# How do AD libraries work?"

# ╔═╡ 685c2b28-b071-452c-a881-801128dcb6c3
md"`ForwardDiff` is operator overloading based, many of its overheads can be optimized by Julia's JIT compiler."

# ╔═╡ 177ddfc2-2cbe-4dba-9d05-2857633dd1ae
md"# [Tapenade](http://tapenade.inria.fr:8080/tapenade/index.jsp)

![](http://tapenade.inria.fr:8080/tapenade/tapenadelogo.gif)"

# ╔═╡ 6c2a3a93-385f-4758-9b6e-4cb594a8e856
md"## Example 1: Bessel Example"

# ╔═╡ fb8168c2-8489-418b-909b-cede57b5ae64
md"bessel.f90"

# ╔═╡ fdb39284-dbb1-49fa-9a1c-f360f9e6b765
md"""
```fortran
subroutine besselj(res, v, z, atol)
    implicit none
	integer, intent(in) :: v
	real*8, intent(in) :: z, atol
	real*8, intent(out) :: res
	real*8 :: s
	integer :: k, i, factv
    k = 0
    factv = 1
    do i = 2,v
        factv = factv * i
    enddo

    s = (z/2.0)**v / factv
    res = s
    do while(abs(s) > atol)
        k = k + 1
        s = -s / k / (k+v) * ((z/2) ** 2)
        res = res + s
    enddo
endsubroutine besselj
```
"""

# ╔═╡ 60214f22-c8bb-4a32-a882-4e6c727b29a9
md"""
besselj_d.f90 (forward mode)
```fortran
!        Generated by TAPENADE     (INRIA, Ecuador team)
!  Tapenade 3.15 (master) - 15 Apr 2020 11:54
!
!  Differentiation of besselj in forward (tangent) mode:
!   variations   of useful results: res
!   with respect to varying inputs: z
!   RW status of diff variables: res:out z:in
SUBROUTINE BESSELJ_D(res, resd, v, z, zd, atol)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: v
  REAL*8, INTENT(IN) :: z, atol
  REAL*8, INTENT(IN) :: zd
  REAL*8, INTENT(OUT) :: res
  REAL*8, INTENT(OUT) :: resd
  REAL*8 :: s
  REAL*8 :: sd
  INTEGER :: k, i, factv
  INTRINSIC ABS
  REAL*8 :: abs0
  REAL*8 :: pwx1
  REAL*8 :: pwx1d
  REAL*8 :: pwr1
  REAL*8 :: pwr1d
  INTEGER :: temp
  k = 0
  factv = 1
  DO i=2,v
    factv = factv*i
  END DO
  pwx1d = zd/2.0
  pwx1 = z/2.0
  IF (pwx1 .LE. 0.0 .AND. (v .EQ. 0.0 .OR. v .NE. INT(v))) THEN
    pwr1d = 0.0_8
  ELSE
    pwr1d = v*pwx1**(v-1)*pwx1d
  END IF
  pwr1 = pwx1**v
  sd = pwr1d/factv
  s = pwr1/factv
  resd = sd
  res = s
  DO WHILE (.true.)
    IF (s .GE. 0.) THEN
      abs0 = s
    ELSE
      abs0 = -s
    END IF
    IF (abs0 .GT. atol) THEN
      k = k + 1
      temp = k*(k+v)*(2*2)
      sd = -((z**2*sd+s*2*z*zd)/temp)
      s = -(s*(z*z)/temp)
      resd = resd + sd
      res = res + s
    ELSE
      EXIT
    END IF
  END DO
END SUBROUTINE BESSELJ_D
```

besselj_b.f90 (backward mode)
```fortran
!        Generated by TAPENADE     (INRIA, Ecuador team)
!  Tapenade 3.15 (master) - 15 Apr 2020 11:54
!
!  Differentiation of besselj in reverse (adjoint) mode:
!   gradient     of useful results: res z
!   with respect to varying inputs: res z
!   RW status of diff variables: res:in-zero z:incr
SUBROUTINE BESSELJ_B(res, resb, v, z, zb, atol)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: v
  REAL*8, INTENT(IN) :: z, atol
  REAL*8 :: zb
  REAL*8 :: res
  REAL*8 :: resb
  REAL*8 :: s
  REAL*8 :: sb
  INTEGER :: k, i, factv
  INTRINSIC ABS
  REAL*8 :: abs0
  REAL*8 :: tempb
  INTEGER :: ad_count
  INTEGER :: i0
  INTEGER :: branch
  k = 0
  factv = 1
  DO i=2,v
    factv = factv*i
  END DO
  s = (z/2.0)**v/factv
  ad_count = 1
  DO WHILE (.true.)
    IF (s .GE. 0.) THEN
      abs0 = s
    ELSE
      abs0 = -s
    END IF
    IF (abs0 .GT. atol) THEN
      CALL PUSHINTEGER4(k)
      k = k + 1
      CALL PUSHREAL8(s)
      s = -(s/k/(k+v)*(z/2)**2)
      ad_count = ad_count + 1
    ELSE
      GOTO 100
    END IF
  END DO
  CALL PUSHCONTROL1B(0)
  GOTO 110
 100 CALL PUSHCONTROL1B(1)
 110 DO i0=1,ad_count
    IF (i0 .EQ. 1) THEN
      CALL POPCONTROL1B(branch)
      IF (branch .EQ. 0) THEN
        sb = 0.0_8
      ELSE
        sb = 0.0_8
      END IF
    ELSE
      sb = sb + resb
      CALL POPREAL8(s)
      tempb = -(sb/(k*(k+v)*2**2))
      sb = z**2*tempb
      zb = zb + 2*z*s*tempb
      CALL POPINTEGER4(k)
    END IF
  END DO
  sb = sb + resb
  IF (.NOT.(z/2.0 .LE. 0.0 .AND. (v .EQ. 0.0 .OR. v .NE. INT(v)))) zb = &
&     zb + v*(z/2.0)**(v-1)*sb/(2.0*factv)
  resb = 0.0_8
END SUBROUTINE BESSELJ_B
```
"""

# ╔═╡ 7a6dbe09-cb7f-405f-b9b5-b350ca170e5f
md"## Example 2: Matrix multiplication"

# ╔═╡ 5dc4a849-76dd-4c4f-8828-755671839e5e
md"""
matmul_b.f90
```fortran
!        Generated by TAPENADE     (INRIA, Ecuador team)
!  Tapenade 3.16 (develop) -  9 Apr 2021 17:40
!
!  Differentiation of mymatmul in reverse (adjoint) mode:
!   gradient     of useful results: x y z
!   with respect to varying inputs: x y z
!   RW status of diff variables: x:incr y:incr z:in-out
SUBROUTINE MYMATMUL_B(z, zb, x, xb, y, yb, m, n, o)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: m, n, o
  REAL*8, DIMENSION(:, :) :: z(m, n)
  REAL*8 :: zb(m, n)
  REAL*8, DIMENSION(:, :), INTENT(IN) :: x(m, o), y(o, n)
  REAL*8 :: xb(m, o), yb(o, n)
  REAL*8 :: temp
  REAL*8 :: tempb
  INTEGER :: i, j, k
  DO j=n,1,-1
    DO i=m,1,-1
      tempb = zb(i, j)
      zb(i, j) = 0.0_8
      DO k=o,1,-1
        xb(i, k) = xb(i, k) + y(k, j)*tempb
        yb(k, j) = yb(k, j) + x(i, k)*tempb
      END DO
    END DO
  END DO
END SUBROUTINE MYMATMUL_B
```
"""

# ╔═╡ b053f11b-9ed7-47ff-ab32-0c70b87e71ed
md"## Example 3: Pyramid"

# ╔═╡ 7b1aa6dd-647f-44cb-b580-b58e23e8b5a6
html"""
<img src="https://user-images.githubusercontent.com/6257240/117090732-228e1a00-ad27-11eb-8231-09c462a17dc7.png" width=500/>
"""

# ╔═╡ b96bac75-b4ad-45f7-aeec-cb6a387eebf0
md"You will see a lot allocation"

# ╔═╡ 5fe022eb-6a17-466e-a6d0-d67e82af23cd
md"pyramid.f90"

# ╔═╡ 92047e95-7eba-4021-9668-9bb4b92261d7
md"""
```fortran
!  Differentiation of pyramid in reverse (adjoint) mode:
!   gradient     of useful results: v x
!   with respect to varying inputs: v x
!   RW status of diff variables: v:in-out x:incr
SUBROUTINE PYRAMID_B(v, vb, x, xb, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL*8 :: v(n, n)
  REAL*8 :: vb(n, n)
  REAL*8, INTENT(IN) :: x(n)
  REAL*8 :: xb(n)
  INTEGER :: i, j
  INTRINSIC SIN
  INTRINSIC COS
  INTEGER :: ad_to
  DO j=1,n
    v(1, j) = x(j)
  END DO
  DO i=1,n-1
    DO j=1,n-i
      CALL PUSHREAL8(v(i+1, j))
      v(i+1, j) = SIN(v(i, j))*COS(v(i, j+1))
    END DO
    CALL PUSHINTEGER4(j - 1)
  END DO
  DO i=n-1,1,-1
    CALL POPINTEGER4(ad_to)
    DO j=ad_to,1,-1
      CALL POPREAL8(v(i+1, j))
      vb(i, j) = vb(i, j) + COS(v(i, j))*COS(v(i, j+1))*vb(i+1, j)
      vb(i, j+1) = vb(i, j+1) - SIN(v(i, j+1))*SIN(v(i, j))*vb(i+1, j)
      vb(i+1, j) = 0.0_8
    END DO
  END DO
  DO j=n,1,-1
    xb(j) = xb(j) + vb(1, j)
    vb(1, j) = 0.0_8
  END DO
END SUBROUTINE PYRAMID_B
```
"""

# ╔═╡ e2ae1084-8759-4f27-8ad1-43a88e434a3d
md"## How does NiLang avoid too many allocation?"

# ╔═╡ edd3aea8-abdb-4e12-9ef9-12ac0fff835b
@i function pyramid!(y!, v!, x::AbstractVector{T}) where T
    @safe @assert size(v!,2) == size(v!,1) == length(x)
    @inbounds for j=1:length(x)
        v![1,j] += x[j]
    end
    @invcheckoff @inbounds for i=1:size(v!,1)-1
        for j=1:size(v!,2)-i
            @routine begin
                @zeros T c s
                c += cos(v![i,j+1])
                s += sin(v![i,j])
            end
            v![i+1,j] += c * s
            ~@routine
        end
    end
    y! += v![end,1]
end

# ╔═╡ a2904efb-186c-449d-b1aa-caf530f88e91
@i function power(x3, x)
	@routine begin
		x2 ← zero(x)
		x2 += x^2
	end
	x3 += x2 * x
	~@routine
end

# ╔═╡ 14faaf82-ad3e-4192-8d48-84adfa30442d
ex = NiLangCore.precom_ex(NiLang, :(for j=1:size(v!,2)-i
            @routine begin
                @zeros T c s
                c += cos(v![i,j+1])
                s += sin(v![i,j])
            end
            v![i+1,j] += c * s
            ~@routine
		end)) |> NiLangCore.rmlines

# ╔═╡ 5d141b88-ec07-4a02-8eb3-37405e5c9f5d
NiLangCore.dual_ex(NiLang, ex)

# ╔═╡ 0907e683-f216-4cf6-a210-ae5181fdc487
function pyramid0!(v!, x::AbstractVector{T}) where T
    @assert size(v!,2) == size(v!,1) == length(x)
    for j=1:length(x)
        v![1,j] = x[j]
    end
    @inbounds for i=1:size(v!,1)-1
        for j=1:size(v!,2)-i
            v![i+1,j] = cos(v![i,j+1]) * sin(v![i,j])
        end
    end
end

# ╔═╡ 0bbfa106-f465-4a7b-80a7-7732ba435822
x = randn(20);

# ╔═╡ 805c7072-98fa-4086-a69d-2e126c55af36
let
	@benchmark pyramid0!(v, x) seconds=1 setup=(x=randn(1000); v=zeros(1000, 1000))
end

# ╔═╡ 7e527024-c294-4c16-8626-9953588d9b6a
let
	@benchmark pyramid!(0.0, v, x) seconds=1 setup=(x=10*randn(1000); v=zeros(1000, 1000))
end

# ╔═╡ 3e59c65a-ceed-42ed-be64-a6964db016e7
pyramid!(0.0, zeros(20, 20), x)

# ╔═╡ 29f85d05-99fd-4843-9be0-5663e681dad7
html"""<img src="https://github.com/GiggleLiu/NiLang.jl/blob/master/examples/pyramid-benchmark.png?raw=true" width=500/>
"""

# ╔═╡ e7830e55-bd9e-4a8a-9239-4191a5f0b1d1
let
	@benchmark NiLang.AD.gradient(Val(1), pyramid!, (0.0, v, x)) seconds=1 setup=(x=randn(1000); v=zeros(1000, 1000))
end

# ╔═╡ de2cd247-ba68-4ba4-9784-27a743478635
md"## NiLang's implementation"

# ╔═╡ dc929c23-7434-4848-847a-9fa696e84776
md"""
```math
\begin{align}
&v_{−1} &= & x_1 &=&1.5000\\
&v_0 &= & x_2 &=&0.5000\\
&v_1 &= & v_{−1}/v_0 &=&1.5000/0.5000 &= 3.0000\\
&v_2 &= & \sin(v1)&=& \sin(3.0000) &= 0.1411\\
&v_3 &= & \exp(v0)&=& \exp(0.5000) &= 1.6487\\
&v_4 &= & v_1 − v_3 &=&3.0000 − 1.6487 &= 1.3513\\
&v_5 &= & v_2 + v_4 &=&0.1411 + 1.3513 &= 1.4924\\
&v_6 &= & v_5 ∗ v_4 &=&1.4924 ∗ 1.3513 &= 2.0167\\
&y &= & v_6 &=&2.0167
\end{align}
```
"""

# ╔═╡ 4f1df03f-c315-47b1-b181-749e1231594c
html"""
<img src="https://user-images.githubusercontent.com/6257240/117074233-168f6180-ad01-11eb-8b16-7ae9836cfdcd.png" width=400/>
"""

# ╔═╡ 7eccba6a-3ad5-440b-9c5d-392dc8dc7aba
@i function example_linear(y::T, x1::T, x2::T) where T
	@routine begin
		@zeros T v1 v2 v3 v4 v5
		v1 += x1 / x2
		v2 += sin(v1)
		v3 += exp(x2)
		v4 += v1 - v3
		v5 += v2 + v4
	end
	y += v5 * v4
	~@routine
end

# ╔═╡ 4a858a3e-ce28-4642-b061-3975a3ed99ff
md"NOTES:
* a statement changes values inplace directly,
* no return statement, returns the input arguments directly
* `@routine <compute>; <copy statements>; ~@routine` is the Bennett's compute copy uncompute design pattern
"

# ╔═╡ 674bb3bb-637b-44f2-bf6d-d1678da03fbd
PlusEq(identity)(2, 3)

# ╔═╡ 5a59d96f-b2f1-4564-82c7-7f0fe181afb8
prettify(@macroexpand @i function f(y::T, x::T) where T
	y.re += x.re
end)

# ╔═╡ 55d2f8ee-4f77-4d44-b704-30643dbbab84
@i function f3(y::T, x::T) where T
	y.re += x.re
end

# ╔═╡ 14951168-97c2-43ae-8d5e-5506408a2bb2
f3(1+2im, 2+3im)

# ╔═╡ 4f564581-6032-449c-8b15-3c741f44237a
x5 = GVar(3+4.0im)

# ╔═╡ a36516e8-76c1-4bff-8a12-3e1e621b857d
~example_linear

# ╔═╡ 402b861c-d363-4d23-b9e9-eb088f57b5c4
expre = NiLangCore.precom_ex(@__MODULE__, :(begin
	@routine begin
		@zeros T v1 v2 v3 v4 v5
		v1 += x1 / x2
		v2 += sin(v1)
		v3 += exp(x2)
		v4 += v1 - v3
		v5 += v2 + v4
	end
	y += v5 * v4
	~@routine
end), NiLangCore.PreInfo(Symbol[])) |> NiLangCore.rmlines

# ╔═╡ 63975a80-1b41-4f55-91a1-4a316ad7bf26
example_linear(0.0, 1.5, 0.5)

# ╔═╡ 6f688f88-432a-42b2-a2db-19d6bb282e0a
NiLangCore.dual_ex(@__MODULE__, expre)

# ╔═╡ fb46db14-f7e0-4f01-9096-02334c62942d
(~example_linear)(example_linear(0.0, 1.5, 0.5)...)

# ╔═╡ b2c3db3d-c250-4daa-8453-3c9a2734aede
md"**How to get gradients?**"

# ╔═╡ 9a986264-5ba7-4697-a00d-711f8efe29f0
let
	y, x1, x2 = 0.0, 1.5, 0.5
	# compute
	(y_out, x1_out, x2_out) = example_linear(y, x1, x2)
	
	# wrap elements with GVar
	y_out_with_g = GVar(y_out, 1.0)
	x1_out_with_g = GVar(x1_out, 0.0)
	x2_out_with_g = GVar(x2_out, 0.0)
	
	# uncompute
	(y_with_g, x1_with_g, x2_with_g) = (~example_linear)(y_out_with_g, x1_out_with_g, x2_out_with_g)
	
	# get gradients
	grad(y_with_g), grad(x1_with_g), grad(x2_with_g)
end

# ╔═╡ 560cf3e9-0c14-4497-85b9-f07045eea32a
with_terminal() do
	dump(GVar)
end

# ╔═╡ 8ab79efc-e8d0-4c6f-81df-a89008142bb7
gvar1 = GVar(1.5, 0.0)

# ╔═╡ 0eec318c-2c09-4dd6-9187-9c0273d29915
grad(gvar1)

# ╔═╡ 1f0ef29c-0ad5-4d97-aeed-5ff44e86577a
gvar2 = GVar(1.0, 2.0)

# ╔═╡ 603d8fc2-5e7b-4d55-92b6-208b25ea6569
grad(gvar2)

# ╔═╡ 2b3c765e-b505-4f07-9bcb-3c8cc47364ad
md"To differentiate operation `y += exp(x)`, we bind the backward rule on its inverse `y -= exp(x)`, i.e. `MinusEq(exp)` in the program."

# ╔═╡ e0f266da-7e65-4398-bfd4-a6c0b54e626b
MinusEq(exp)(gvar2, gvar1)

# ╔═╡ e1d35886-79d0-40a5-bd33-1c4e5f4a0a9a
md"""
```math
\left(\begin{matrix}\overline y& \overline x\end{matrix}\right) \rightarrow \left(\begin{matrix}\overline y& \overline x\end{matrix}\right)\left(\begin{matrix}
1 & \exp(x) \\
0 & 1
\end{matrix}\right) = \left(\begin{matrix}\overline y& \overline x + \exp(x) \overline y\end{matrix}\right)
```
"""

# ╔═╡ b63a30b0-c75b-4998-a2b2-0b79574cab81
exp(1.5) * 2

# ╔═╡ 139bf020-c4a8-45c8-96fa-aeebc7ddaedc
md"*one line version*"

# ╔═╡ 8967c0f0-89f8-4893-b11b-253333d1a823
NiLang.AD.gradient(example_linear, (0.0, 1.5, 0.5); iloss=1)

# ╔═╡ f2540450-5a07-4fb8-93fb-a6d48dd36a56
md"## Control Flows"

# ╔═╡ 3acb2cfd-fa29-4a2b-8f23-f5aaf474edd0
(@code_julia for i=1:10
	x += y
end) |> NiLangCore.rmlines

# ╔═╡ aa1547f2-5edd-4b7e-b93e-bdfc4e4fc6d5
md"""# Memory Management"""

# ╔═╡ 6e76a107-4f51-4e32-b133-7b6e04d7d107
md"The true reverse mode autodiff has to handle the memory wall problem."

# ╔═╡ 999f7a8f-d72e-4ccd-8cbf-b5bbb7db1842
md"""
## Checkpointing
"""

# ╔═╡ 32772c2a-6b80-4779-963c-06974ff0d832
html"""
<img src="https://raw.githubusercontent.com/GiggleLiu/WuLiXueBao/master/paper/tikzimg-1.svg" style="clip-path: inset(0px 300px 40px 0px); margin-left:40px;" width=600/>
"""

# ╔═╡ 41642bd5-1321-490a-95ad-4c1d6363456f
md"
* red arrow: back propagation
* black dot: cached
* white dot: not cached
"

# ╔═╡ 2a553e32-05ef-4c2d-aba7-41185c6035d4
md"Most time efficient (checkpoint every step)"

# ╔═╡ ab8345ce-e038-4d6b-9e1f-57e4f33bb67b
html"""
<img src="https://raw.githubusercontent.com/GiggleLiu/WuLiXueBao/master/paper/tikzimg3-1.svg" style="clip-path: inset(0px 0px 0px 0px); margin-left:40px;" width=300/>
"""

# ╔═╡ bb9c9a4c-601a-4708-9b2d-04d1583938f2
md"Most space efficient (only checkpoint the first step)"

# ╔═╡ b9917e94-c33d-423f-a478-3252bacc2494
html"""
<img src="https://raw.githubusercontent.com/GiggleLiu/WuLiXueBao/master/paper/tikzimg4-1.svg" style="clip-path: inset(0px 0px 0px 0px); margin-left:40px;" width=300/>
"""

# ╔═╡ 4978f404-11ff-41b8-a673-f2d051b1f526
md"Restricting the number of checkpoints, is evenly checkpointed program optimal?"

# ╔═╡ 73bd2e3b-902f-461b-860f-246257608ecd
html"""
<img src="https://raw.githubusercontent.com/GiggleLiu/WuLiXueBao/master/paper/tikzimg2-1.svg" style="clip-path: inset(0px 0px 0px 0px); margin-left:40px;" width=500/>
"""

# ╔═╡ 4dd47dc8-6dfa-47a4-a088-689b4b870762
md"## Optimal checkpointing"

# ╔═╡ ecd975d2-9374-4f40-80ac-2cceda11e7fb
md"""
1992 ~ Andreas Griewank, Achieving logarithmic growth of temporal and spatial complexity in reverse automatic differentiation.

Julia implementation: [TreeverseAlgorithm.jl](https://github.com/GiggleLiu/TreeverseAlgorithm.jl)
"""

# ╔═╡ 832cc81d-a49d-46e7-9d2b-d8bde9bb1273
html"""
<img src="https://user-images.githubusercontent.com/6257240/116494309-91263000-a86e-11eb-8054-9b91646be0e5.png" style="clip-path: inset(74px 350px 0px 0px);"/>
"""

# ╔═╡ 2192a1de-1042-4b13-a313-b67de489124c
md"""
1. Devide the program into ``\delta`` segments, each segment having size $\eta(\delta, \tau) = \frac{(\delta+\tau)!}{\delta! \tau!}$, where ``\delta=1,...,d`` and ``\tau=t-1``.
2. Cache the first state of each segment,
3. Compute gradients in the last segment,
4. Deallocate last checkpoint,
5. Devide the second last segments into two parts.
6. Recursively apply treeverse (Step 2-5).
"""

# ╔═╡ 01c709c7-806c-4389-bbb2-4081e64426d9
md"total number of steps ``T = \eta(d, t)``, both ``t`` and ``d`` can be logarithmic"

# ╔═╡ b1e0cf83-4337-4044-a7d1-5fca8ae79268
md"## An example"

# ╔═╡ 71f4b476-027d-4c8f-b561-1ee418bc9e61
html"""
<img src="https://raw.githubusercontent.com/GiggleLiu/WuLiXueBao/master/paper/bennett_treeverse_pebbles.svg" style="clip-path: inset(50px 350px 0px 0px);"/>
"""

# ╔═╡ 042013cf-9cd2-409d-827f-a311a2f8ce62
md"""
* black dot: current step,
* gray dot: checkpointed state,
* empty dot: state deallocated in current step,
* red square: gradient computed.
"""

# ╔═╡ 82593cd0-1403-4597-8370-919c80494479
md"# Program is not always linear!"

# ╔═╡ f58720b5-2bcb-4950-b453-bd59f648c66a
md"You think your program is like"

# ╔═╡ 4576d791-6af7-4ba5-9b80-fe99c0bb2e88
let
	Compose.set_default_graphic_size(15cm, 3cm)
	nb = nodestyle(:circle, r=0.01)
	eb = compose(context(), bondstyle(:default, r=0.1), Compose.arrow(), linewidth(0.2mm))
	loc(i) = (i/11, 0.5)
	eloc(i) = (loc(i-1) .- (-0.02, 0.0), loc(i) .- (0.025, 0.0))
	canvas() do
		for i=1:10
			nb >> loc(i)
			i == 1 || eb >> eloc(i)
		end
	end
end

# ╔═╡ 6e9d17f1-b17d-4e8d-82a3-921558a20c0f
md"or a DAG (directed acyclic graph)"

# ╔═╡ f18d89f5-1129-43e0-8b4a-5c1fcd618eab
let
	Compose.set_default_graphic_size(15cm, 3cm)
	nb = nodestyle(:circle, r=0.01)
	eb = compose(context(), bondstyle(:default, r=0.1), Compose.arrow(), linewidth(0.2mm))
	loc(i) = (i/11, 0.2)
	loc2(i) = (i/11, 0.7)
	eloc(i, j) = shrink(loc(i), loc(j), 0.02, 0.025)
	eloc2(i, j) = shrink(loc2(i), loc2(j), 0.02, 0.025)
	eloc12(i, j) = shrink(loc(i), loc2(j), 0.1, 0.15)
	eloc21(i, j) = shrink(loc2(i), loc(j), 0.05, 0.1)
	canvas() do
		for i=1:10
			nb >> loc(i)
			i == 1 || eb >> eloc(i-1,i)
		end
		for i=2:5
			nb >> loc2(i)
			i == 2 || eb >> eloc2(i-1, i)
		end
		eb >> eloc12(2,2)
		eb >> eloc12(4,5)
		eb >> eloc21(5,7)
	end
end

# ╔═╡ 2912c7ed-75e3-4dfd-9c40-92115cc08194
md"The truth is"

# ╔═╡ 5d1517c0-562b-40db-bec2-32b5494de1b8
let
	Compose.set_default_graphic_size(15cm, 3cm)
	nb = nodestyle(:circle, r=0.01)
	tb = textstyle(:default)
	eb = compose(context(), bondstyle(:default, r=0.1), Compose.arrow(), linewidth(0.2mm))
	eb2 = compose(context(), bondstyle(:dcurve, r=0.8), Compose.arrow(), linewidth(0.2mm))
	loc(i) = (i/11, 0.2)
	loc2(i) = (i/11, 0.7)
	eloc(i, j) = shrink(loc(i), loc(j), 0.02, 0.025)
	eloc2(i, j) = shrink(loc2(j), loc2(i), 0.02, 0.025)
	eloc12(i, j) = shrink(loc2(j), loc(i), 0.1, 0.15)
	eloc21(i, j) = shrink(loc(j), loc2(i), 0.05, 0.1)
	canvas() do
		for i=1:10
			nb >> loc(i)
			i == 1 || eb >> eloc(i-1,i)
		end
		for i=2:5
			nb >> loc2(i)
			i == 2 || eb >> eloc2(i-1, i)
		end
		eb >> eloc12(2,2)
		eb >> eloc12(4,5)
		tb >> ((0.3, 0.45), "× n")
		
		for i=7:8
			nb >> loc2(i)
			i == 7 || eb >> eloc2(i-1, i)
		end
		eb >> eloc12(7,7)
		eb >> eloc12(8,8)
		tb >> ((0.68, 0.45), "× ∞")
		
		eb2 >> (loc(6) .+ (0.0, 0.1), loc(9) .+ (0, 0.15))
	end
end

# ╔═╡ ae096ad2-3ae9-4440-a959-0d7d9a174f1d
md"## Example 3: Sparse matrix multiplication"

# ╔═╡ 8148bc1f-ef99-40a4-a5ce-0a42643f703d
md"original implementation: [https://github.com/JuliaLang/julia/blob/master/stdlib/SparseArrays/src/linalg.jl](https://github.com/JuliaLang/julia/blob/master/stdlib/SparseArrays/src/linalg.jl)
"

# ╔═╡ bd86c5c2-16be-4cfd-ba7a-a0e2544d82d1
@i function mul!(C::StridedVecOrMat{T}, A::SparseMatrixCSC{T}, B::StridedVecOrMat{T}, α::Number) where T
    @safe A.n == size(B, 1) || throw(DimensionMismatch())
    @safe A.m == size(C, 1) || throw(DimensionMismatch())
    @safe size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    @invcheckoff for k = 1:size(C, 2)
        @inbounds for col = 1:A.n
            @routine begin
                αxj ← zero(T)
                αxj += α*B[col,k]
            end
            for j = A.colptr[col]:(A.colptr[col + 1] - 1)
                C[A.rowval[j], k] += A.nzval[j]*αxj
            end
            ~@routine
        end
    end
end

# ╔═╡ 11557d6b-3a1e-416d-874f-b8d217976f76
md"## Example 4: How to differentiate QR"

# ╔═╡ 48a10ea2-5d32-4a55-b8c0-f6a5e82eace9
md"original implementation: [https://github.com/JuliaLang/julia/blob/master/stdlib/LinearAlgebra/src/qr.jl](https://github.com/JuliaLang/julia/blob/master/stdlib/LinearAlgebra/src/qr.jl)
"

# ╔═╡ fafc1b0f-6469-4b6c-a00d-5272a45fc69b
md"See also"

# ╔═╡ ad6cff7b-5cbf-4ab1-94f7-d21cbc171000
leftright(html"<img src='https://images-na.ssl-images-amazon.com/images/I/41JjpllrDrL._SX364_BO1,204,203,200_.jpg' width=150/>", md"**Matrix computations**
	
Golub, Gene H., and Charles F. Van Loan (2013)")

# ╔═╡ 4d373cf6-9b39-44bc-8f13-220933fc8f5c
function qrfactPivotedUnblocked!(A::AbstractMatrix)
    m, n = size(A)
    piv = Vector(UnitRange{BlasInt}(1,n))
    τ = Vector{eltype(A)}(undef, min(m,n))
    for j = 1:min(m,n)

        # Find column with maximum norm in trailing submatrix
        jm = indmaxcolumn(view(A, j:m, j:n)) + j - 1

        if jm != j
            # Flip elements in pivoting vector
            tmpp = piv[jm]
            piv[jm] = piv[j]
            piv[j] = tmpp

            # Update matrix with
            for i = 1:m
                tmp = A[i,jm]
                A[i,jm] = A[i,j]
                A[i,j] = tmp
            end
        end

        # Compute reflector of columns j
        x = view(A, j:m, j)
        τj = LinearAlgebra.reflector!(x)
        τ[j] = τj

        # Update trailing submatrix with reflector
        LinearAlgebra.reflectorApply!(x, τj, view(A, j:m, j+1:n))
    end
    return LinearAlgebra.QRPivoted{eltype(A), typeof(A)}(A, τ, piv)
end

# ╔═╡ 293a68ca-e02f-47b3-85ed-aeeb8995f3ec
struct Reflector{T,RT,VT<:AbstractVector{T}}
    ξ::T
    normu::RT
    sqnormu::RT
    r::T
    y::VT
end

# ╔═╡ fa5716f9-8bff-4295-812b-691ccdc12832
struct QRPivotedRes{T,RT,VT}
    factors::Matrix{T}
    τ::Vector{T}
    jpvt::Vector{Int}
    reflectors::Vector{Reflector{T,RT,VT}}
    vAs::Vector{Vector{T}}
    jms::Vector{Int}
end

# ╔═╡ 8324f365-fd12-4ca3-8ca6-657e5917f946
# Elementary reflection similar to LAPACK. The reflector is not Hermitian but
# ensures that tridiagonalization of Hermitian matrices become real. See lawn72
@i function reflector!(R::Reflector{T,RT}, x::AbstractVector{T}) where {T,RT}
    n ← length(x)
    @inbounds @invcheckoff if n != 0
        @zeros T ξ1
        @zeros RT normu sqnormu
        ξ1 += x[1]
        sqnormu += abs2(ξ1)
        for i = 2:n
            sqnormu += abs2(x[i])
        end
        if !iszero(sqnormu)
            normu += sqrt(sqnormu)
            if real(ξ1) < 0
                NEG(normu)
            end
            ξ1 += normu
            R.y[1] -= normu
            for i = 2:n
                R.y[i] += x[i] / ξ1
            end
            R.r += ξ1/normu
        end
        SWAP(R.ξ, ξ1)
        SWAP(R.normu, normu)
        SWAP(R.sqnormu, sqnormu)
    end
end

# ╔═╡ 70fb10ea-9229-46ef-8ba3-b1d3874b7929
# apply reflector from left
@i function reflectorApply!(vA::AbstractVector{T}, x::AbstractVector, τ::Number, A::StridedMatrix{T}) where T
    (m, n) ← size(A)
    if length(x) != m || length(vA) != n
        @safe throw(DimensionMismatch("reflector has length ($(length(x)), $(length(vA))), which must match the first dimension of matrix A, ($m, $n)"))
    end
    @inbounds @invcheckoff if m != 0
        for j = 1:n
            # dot
            @zeros T vAj vAj_τ
            vAj += A[1, j]
            for i = 2:m
                vAj += x[i]'*A[i, j]
            end
            vAj_τ += τ' * vAj
            # ger
            A[1, j] -= vAj_τ
            for i = 2:m
                A[i, j] -= x[i]*vAj_τ
            end
            vAj_τ -= τ' * vAj
            SWAP(vA[j], vAj)
        end
    end
end

# ╔═╡ 51504ba4-4711-48b7-aab9-d4f26c009659
function alloc(::typeof(reflector!), x::AbstractVector{T}) where T
	RT = real(T)
	Reflector(zero(T), zero(RT), zero(RT), zero(T), zero(x))
end

# ╔═╡ f267e315-3c19-4345-8fba-641bb0ea515b
@i function qr_pivoted!(res::QRPivotedRes, A::StridedMatrix{T}) where T
    m, n ← size(A)
    @invcheckoff @inbounds for j = 1:min(m,n)
        # Find column with maximum norm in trailing submatrix
        jm ← LinearAlgebra.indmaxcolumn(NiLang.value.(view(A, j:m, j:n))) + j - 1

        if jm != j
            # Flip elements in pivoting vector
            SWAP(res.jpvt[jm], res.jpvt[j])

            # Update matrix with
            for i = 1:m
                SWAP(A[i, jm], A[i, j])
            end
        end

        # Compute reflector of columns j
        R ← alloc(reflector!, A |> subarray(j:m, j))
        vA ← zeros(T, n-j)
        reflector!(R, A |> subarray(j:m, j))
        # Update trailing submatrix with reflector
        reflectorApply!(vA, R.y, R.r, A |> subarray(j:m, j+1:n))
        for i=1:length(R.y)
            SWAP(R.y[i], A[j+i-1, j])
        end
        PUSH!(res.reflectors, R)
        PUSH!(res.vAs, vA)
        PUSH!(res.jms, jm)
        R → _zero(Reflector{T,real(T),Vector{T}})
        vA → zeros(T, 0)
        jm → 0
    end
    @inbounds for i=1:length(res.reflectors)
        res.τ[i] += res.reflectors[i].r
    end
    res.factors += A
end

# ╔═╡ a07b93b1-742b-41d4-bd0f-bc899de55338
function alloc_qr(A::AbstractMatrix{T}) where T
	(m, n) = size(A)
	τ = zeros(T, min(m,n))
	jpvt = collect(1:n)
	reflectors = Reflector{T,real(T),Vector{T}}[]
	vAs = Vector{T}[]
	jms = Int[]
	QRPivotedRes(zero(A), τ, jpvt, reflectors, vAs, jms)
end

# ╔═╡ 5f207f59-b9f4-477f-b79f-0aee743bdb8e
A = randn(ComplexF64, 20, 20);

# ╔═╡ f88517d6-b87d-45ba-bf3f-67074fa51fca
@test qr_pivoted!(alloc_qr(A), copy(A))[1].factors ≈ LinearAlgebra.qrfactPivotedUnblocked!(copy(A)).factors

# ╔═╡ 45aef837-9b2c-49b2-b815-e4d60f103f58
let
	@testset "qr pivoted gradient" begin
		# rank deficient initial matrix
		n = 50
		U = LinearAlgebra.qr(randn(n, n)).Q
		Σ = Diagonal((x=randn(n); x[n÷2+1:end] .= 0; x))
		A = U*Σ*U'
		res = alloc_qr(A)
		@test rank(A) == n ÷ 2
		qrres = qr_pivoted!(deepcopy(res), copy(A))[1]
		@test count(x->(x>1e-12), sum(abs2, QRPivoted(qrres.factors, qrres.τ, qrres.jpvt).R, dims=2)) == n ÷ 2

		@i function loss(y, qrres, A)
			qr_pivoted!(qrres, A)
			y += abs(qrres.factors[1])
		end
		nrloss(A) = loss(0.0, deepcopy(res), A)[1]
		ngA = zero(A)
		δ = 1e-5
		for j=1:size(A, 2)
			for i=1:size(A, 1)
				A_ = copy(A)
				A_[i,j] -= δ/2
				l1 = nrloss(copy(A_))
				A_[i,j] += δ
				l2 = nrloss(A_)
				ngA[i,j] = (l2-l1)/δ
			end
		end
		gA = NiLang.AD.gradient(loss, (0.0, res, A); iloss=1)[3]
		@test real.(gA) ≈ ngA
	end
end

# ╔═╡ Cell order:
# ╟─a1ef579e-4b66-4042-944e-7e27c660095e
# ╟─100b4293-fd1e-4b9c-a831-5b79bc2a5ebe
# ╟─f11023e5-8f7b-4f40-86d3-3407b61863d9
# ╟─9d11e058-a7d0-11eb-1d78-6592ff7a1b43
# ╟─b73157bf-1a77-47b8-8a06-8d6ec2045023
# ╟─ec13e0a9-64ff-4f66-a5a6-5fef53428fa1
# ╟─f8b0d1ce-99f7-4729-b46e-126da540cbbe
# ╟─435ac19e-1c0c-4ee5-942d-f2a97c8c4d80
# ╟─48ecd619-d01d-43ff-8b52-7c2566c3fa2b
# ╟─4878ce45-40ff-4fae-98e7-1be41e930e4d
# ╠═ce44f8bd-692e-4eab-9ba4-055b25e40c81
# ╠═b2c1936c-2c27-4fbb-8183-e38c5e858483
# ╠═8be1b812-fcac-404f-98aa-0571cb990f34
# ╟─33e0c762-c75e-44aa-bfe2-bff92dd1ace8
# ╟─c59c35ee-1907-4736-9893-e22c052150ca
# ╠═0ae13734-b826-4dbf-93d1-11044ce88bd4
# ╠═99187515-c8be-49c2-8d70-9c2998d9993c
# ╟─78ca6b08-84c4-4e4d-8412-ae6c28bfafce
# ╠═f12b25d8-7c78-4686-b46d-00b34e565605
# ╟─d90c3cc9-084d-4cf7-9db7-42cea043030b
# ╟─93c98cb2-18af-47df-afb3-8c5a34b4723c
# ╟─2dc74e15-e2ea-4961-b43f-0ada1a73d80a
# ╟─7ee75a15-eaea-462a-92b6-293813d2d4d7
# ╟─02a25b73-7353-43b1-8738-e7ca472d0cc7
# ╟─2afb984f-624e-4381-903f-ccc1d8a66a17
# ╟─7e5d5e69-90f2-4106-8edf-223c150a8168
# ╟─92d7a938-9463-4eee-8839-0b8c5f762c79
# ╟─4b1a0b59-ddc6-4b2d-b5f5-d92084c31e46
# ╟─81f16b8b-2f0b-4ba3-8c26-6669eabf48aa
# ╟─fb6c3a48-550a-4d2e-a00b-a1e40d86b535
# ╟─ab6fa4ac-29ed-4722-88ed-fa1caf2072f3
# ╟─8e72d934-e307-4505-ac82-c06734415df6
# ╟─e6ff86a9-9f54-474b-8111-a59a25eda506
# ╟─9c1d9607-a634-4350-aacd-2d40984d647d
# ╟─63db2fa2-50b2-4940-b8ee-0dc6e3966a57
# ╟─693167e7-e80c-401d-af89-55b5fae30848
# ╟─4cd70901-2142-4868-9a33-c46ca0d064ec
# ╟─89018a35-76f4-4f23-b15a-a600db046d6f
# ╟─1d219222-0778-4c37-9182-ed5ccbb3ef32
# ╟─4ff09f7c-aeac-48bd-9d58-8446137c3acd
# ╟─ea44037b-9359-4fbd-990f-529d88d54351
# ╟─e731a8e3-6462-4a60-83e9-6ab7ddfff50e
# ╟─685c2b28-b071-452c-a881-801128dcb6c3
# ╟─177ddfc2-2cbe-4dba-9d05-2857633dd1ae
# ╟─6c2a3a93-385f-4758-9b6e-4cb594a8e856
# ╟─fb8168c2-8489-418b-909b-cede57b5ae64
# ╟─fdb39284-dbb1-49fa-9a1c-f360f9e6b765
# ╟─60214f22-c8bb-4a32-a882-4e6c727b29a9
# ╟─7a6dbe09-cb7f-405f-b9b5-b350ca170e5f
# ╟─5dc4a849-76dd-4c4f-8828-755671839e5e
# ╟─b053f11b-9ed7-47ff-ab32-0c70b87e71ed
# ╟─7b1aa6dd-647f-44cb-b580-b58e23e8b5a6
# ╟─b96bac75-b4ad-45f7-aeec-cb6a387eebf0
# ╟─5fe022eb-6a17-466e-a6d0-d67e82af23cd
# ╟─92047e95-7eba-4021-9668-9bb4b92261d7
# ╟─e2ae1084-8759-4f27-8ad1-43a88e434a3d
# ╠═edd3aea8-abdb-4e12-9ef9-12ac0fff835b
# ╠═a2904efb-186c-449d-b1aa-caf530f88e91
# ╠═14faaf82-ad3e-4192-8d48-84adfa30442d
# ╠═5d141b88-ec07-4a02-8eb3-37405e5c9f5d
# ╠═0907e683-f216-4cf6-a210-ae5181fdc487
# ╠═805c7072-98fa-4086-a69d-2e126c55af36
# ╠═7e527024-c294-4c16-8626-9953588d9b6a
# ╠═0bbfa106-f465-4a7b-80a7-7732ba435822
# ╠═3e59c65a-ceed-42ed-be64-a6964db016e7
# ╟─29f85d05-99fd-4843-9be0-5663e681dad7
# ╠═9a46597c-b1ee-4e3b-aed1-fd2874b6e77a
# ╠═e7830e55-bd9e-4a8a-9239-4191a5f0b1d1
# ╟─de2cd247-ba68-4ba4-9784-27a743478635
# ╟─dc929c23-7434-4848-847a-9fa696e84776
# ╟─4f1df03f-c315-47b1-b181-749e1231594c
# ╠═ccd38f52-104d-434a-aea3-dd94e571374f
# ╠═7eccba6a-3ad5-440b-9c5d-392dc8dc7aba
# ╠═f4230251-ba54-434a-b86b-f972c7389217
# ╟─4a858a3e-ce28-4642-b061-3975a3ed99ff
# ╠═674bb3bb-637b-44f2-bf6d-d1678da03fbd
# ╠═5a59d96f-b2f1-4564-82c7-7f0fe181afb8
# ╠═55d2f8ee-4f77-4d44-b704-30643dbbab84
# ╠═14951168-97c2-43ae-8d5e-5506408a2bb2
# ╠═4f564581-6032-449c-8b15-3c741f44237a
# ╠═a36516e8-76c1-4bff-8a12-3e1e621b857d
# ╠═402b861c-d363-4d23-b9e9-eb088f57b5c4
# ╠═63975a80-1b41-4f55-91a1-4a316ad7bf26
# ╠═6f688f88-432a-42b2-a2db-19d6bb282e0a
# ╠═fb46db14-f7e0-4f01-9096-02334c62942d
# ╟─b2c3db3d-c250-4daa-8453-3c9a2734aede
# ╠═69dc2685-b70f-4a81-af30-f02e0054bd52
# ╠═9a986264-5ba7-4697-a00d-711f8efe29f0
# ╠═560cf3e9-0c14-4497-85b9-f07045eea32a
# ╠═8ab79efc-e8d0-4c6f-81df-a89008142bb7
# ╠═0eec318c-2c09-4dd6-9187-9c0273d29915
# ╠═1f0ef29c-0ad5-4d97-aeed-5ff44e86577a
# ╠═603d8fc2-5e7b-4d55-92b6-208b25ea6569
# ╟─2b3c765e-b505-4f07-9bcb-3c8cc47364ad
# ╠═e0f266da-7e65-4398-bfd4-a6c0b54e626b
# ╟─e1d35886-79d0-40a5-bd33-1c4e5f4a0a9a
# ╠═b63a30b0-c75b-4998-a2b2-0b79574cab81
# ╟─139bf020-c4a8-45c8-96fa-aeebc7ddaedc
# ╠═8967c0f0-89f8-4893-b11b-253333d1a823
# ╟─f2540450-5a07-4fb8-93fb-a6d48dd36a56
# ╠═3acb2cfd-fa29-4a2b-8f23-f5aaf474edd0
# ╟─aa1547f2-5edd-4b7e-b93e-bdfc4e4fc6d5
# ╟─6e76a107-4f51-4e32-b133-7b6e04d7d107
# ╟─999f7a8f-d72e-4ccd-8cbf-b5bbb7db1842
# ╟─32772c2a-6b80-4779-963c-06974ff0d832
# ╟─41642bd5-1321-490a-95ad-4c1d6363456f
# ╟─2a553e32-05ef-4c2d-aba7-41185c6035d4
# ╟─ab8345ce-e038-4d6b-9e1f-57e4f33bb67b
# ╟─bb9c9a4c-601a-4708-9b2d-04d1583938f2
# ╟─b9917e94-c33d-423f-a478-3252bacc2494
# ╟─4978f404-11ff-41b8-a673-f2d051b1f526
# ╟─73bd2e3b-902f-461b-860f-246257608ecd
# ╟─4dd47dc8-6dfa-47a4-a088-689b4b870762
# ╟─ecd975d2-9374-4f40-80ac-2cceda11e7fb
# ╟─832cc81d-a49d-46e7-9d2b-d8bde9bb1273
# ╟─2192a1de-1042-4b13-a313-b67de489124c
# ╟─01c709c7-806c-4389-bbb2-4081e64426d9
# ╟─b1e0cf83-4337-4044-a7d1-5fca8ae79268
# ╟─71f4b476-027d-4c8f-b561-1ee418bc9e61
# ╟─042013cf-9cd2-409d-827f-a311a2f8ce62
# ╟─82593cd0-1403-4597-8370-919c80494479
# ╟─f58720b5-2bcb-4950-b453-bd59f648c66a
# ╟─4576d791-6af7-4ba5-9b80-fe99c0bb2e88
# ╟─6e9d17f1-b17d-4e8d-82a3-921558a20c0f
# ╟─f18d89f5-1129-43e0-8b4a-5c1fcd618eab
# ╟─2912c7ed-75e3-4dfd-9c40-92115cc08194
# ╟─5d1517c0-562b-40db-bec2-32b5494de1b8
# ╟─ae096ad2-3ae9-4440-a959-0d7d9a174f1d
# ╟─8148bc1f-ef99-40a4-a5ce-0a42643f703d
# ╠═200f1848-0980-4185-919a-93ab2e7f788f
# ╠═bd86c5c2-16be-4cfd-ba7a-a0e2544d82d1
# ╟─11557d6b-3a1e-416d-874f-b8d217976f76
# ╟─48a10ea2-5d32-4a55-b8c0-f6a5e82eace9
# ╟─fafc1b0f-6469-4b6c-a00d-5272a45fc69b
# ╟─ad6cff7b-5cbf-4ab1-94f7-d21cbc171000
# ╠═30c191c5-642b-4062-98f3-643d314a054d
# ╠═fa5716f9-8bff-4295-812b-691ccdc12832
# ╠═f267e315-3c19-4345-8fba-641bb0ea515b
# ╠═4d373cf6-9b39-44bc-8f13-220933fc8f5c
# ╠═293a68ca-e02f-47b3-85ed-aeeb8995f3ec
# ╠═8324f365-fd12-4ca3-8ca6-657e5917f946
# ╠═70fb10ea-9229-46ef-8ba3-b1d3874b7929
# ╠═51504ba4-4711-48b7-aab9-d4f26c009659
# ╠═a07b93b1-742b-41d4-bd0f-bc899de55338
# ╠═864dbde7-b689-4165-a08e-6bbbd72190de
# ╠═5f207f59-b9f4-477f-b79f-0aee743bdb8e
# ╠═f88517d6-b87d-45ba-bf3f-67074fa51fca
# ╠═45aef837-9b2c-49b2-b815-e4d60f103f58

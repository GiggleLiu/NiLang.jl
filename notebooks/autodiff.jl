### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# ╔═╡ f11023e5-8f7b-4f40-86d3-3407b61863d9
using PlutoUI, Viznet, Compose, Plots

# ╔═╡ ce44f8bd-692e-4eab-9ba4-055b25e40c81
using ForwardDiff: Dual

# ╔═╡ a1ef579e-4b66-4042-944e-7e27c660095e
md"""
```math
\newcommand{\comment}[1]{{\bf  \color{blue}{\text{◂~ #1}}}}
```
"""

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
        * primitves on tensors (including tensorflow, pytorch et al.)
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

# ╔═╡ f12b25d8-7c78-4686-b46d-00b34e565605
let
	x = Dual(π/4, 1.0)
	for i=1:10
		x = sin(x)
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
		lb >> ((fi0 .+ x) ./ 2 .- (0.08, 0.45), raw"\color{red}{\frac{\partial \vec{y}_{i}}{\partial x}}")
		lb >> ((fi0 .+ fi1) ./ 2 .- (0.08, 0.45), raw"\color{red}{\frac{\partial \vec{y}_{i+1}}{\partial x}}")
		ebd >> (fi1, fi2)
		eb >> (fi0, fi1)
		eb >> (x, fi0)
		#arrow((fi1 .+ fi0) ./ 2 .+ (0.08, -0.3), (fi1 .+ fi2) ./ 2 .+ (-0.08, -0.3))
		arrow( (fi0 .+ fi1) ./ 2 .+ (-0.08, -0.3), (fi0 .+ x) ./ 2 .+ (0.05, -0.3),)
	end
	img
end

# ╔═╡ 4b1a0b59-ddc6-4b2d-b5f5-d92084c31e46
md"### How to visite local Jacobians in the reversed order? "

# ╔═╡ a7fc71a2-6d45-4162-8073-4ddb85ded2e8
md"
**Design Decision**

1. Compute forward pass and caching inetermediate results into a global stack $\Sigma$ （packages except NiLang），
2. reversible programming."

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
    * reverse the program tape by **caching/checkpointing** v.s. **reversible programming**
4. Julia has one of the most active AD community!

#### Forward v.s. Backward
when is forward mode AD more useful?

* It is often combined with backward mode AD for obtaining Hessians (forward over backward).
* Having <20 input parameters.

when is backward mode AD more useful?
* In most variational optimizations, especially when we are training a neural network with ~ 100M parameters.
"

# ╔═╡ aa1547f2-5edd-4b7e-b93e-bdfc4e4fc6d5
md"""# The "true" reverse mode automatic differentiation"""

# ╔═╡ 6e76a107-4f51-4e32-b133-7b6e04d7d107
md"The true reverse mode autodiff has to handle the memory wall problem."

# ╔═╡ 32772c2a-6b80-4779-963c-06974ff0d832
html"""
<img src="https://raw.githubusercontent.com/GiggleLiu/WuLiXueBao/master/paper/tikzimg-1.svg" style="clip-path: inset(0px 230px 0px 0px); margin-left:40px;" width=500/>
"""

# ╔═╡ 999f7a8f-d72e-4ccd-8cbf-b5bbb7db1842
md"""
#### The optimal checkpointing

1992 ~ Andreas Griewank, Achieving logarithmic growth of temporal and spatial complexity in reverse automatic differentiation.

Julia implementation: [TreeverseAlgorithm.jl](https://github.com/GiggleLiu/TreeverseAlgorithm.jl)
"""

# ╔═╡ 832cc81d-a49d-46e7-9d2b-d8bde9bb1273
html"""
<img src="https://user-images.githubusercontent.com/6257240/116494309-91263000-a86e-11eb-8054-9b91646be0e5.png" style="clip-path: inset(40px 350px 0px 0px);"/>
"""

# ╔═╡ 71f4b476-027d-4c8f-b561-1ee418bc9e61
html"""
<img src="https://raw.githubusercontent.com/GiggleLiu/WuLiXueBao/master/paper/bennett_treeverse_pebbles.svg" style="clip-path: inset(50px 350px 0px 0px);"/>
"""

# ╔═╡ 82593cd0-1403-4597-8370-919c80494479
md"# Our program is not linear!"

# ╔═╡ 6bf46802-0586-42f8-bd7e-9f0c5a36689b
md"# Examples"

# ╔═╡ ae096ad2-3ae9-4440-a959-0d7d9a174f1d
md"## How to differentiate sparse matrix multiplication"

# ╔═╡ 11557d6b-3a1e-416d-874f-b8d217976f76
md"## How to differentiate QR"

# ╔═╡ Cell order:
# ╟─a1ef579e-4b66-4042-944e-7e27c660095e
# ╠═f11023e5-8f7b-4f40-86d3-3407b61863d9
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
# ╟─a7fc71a2-6d45-4162-8073-4ddb85ded2e8
# ╟─fb6c3a48-550a-4d2e-a00b-a1e40d86b535
# ╟─ab6fa4ac-29ed-4722-88ed-fa1caf2072f3
# ╟─8e72d934-e307-4505-ac82-c06734415df6
# ╟─e6ff86a9-9f54-474b-8111-a59a25eda506
# ╟─9c1d9607-a634-4350-aacd-2d40984d647d
# ╟─63db2fa2-50b2-4940-b8ee-0dc6e3966a57
# ╟─693167e7-e80c-401d-af89-55b5fae30848
# ╟─4cd70901-2142-4868-9a33-c46ca0d064ec
# ╟─4ff09f7c-aeac-48bd-9d58-8446137c3acd
# ╟─ea44037b-9359-4fbd-990f-529d88d54351
# ╟─aa1547f2-5edd-4b7e-b93e-bdfc4e4fc6d5
# ╟─6e76a107-4f51-4e32-b133-7b6e04d7d107
# ╟─32772c2a-6b80-4779-963c-06974ff0d832
# ╟─999f7a8f-d72e-4ccd-8cbf-b5bbb7db1842
# ╟─832cc81d-a49d-46e7-9d2b-d8bde9bb1273
# ╟─71f4b476-027d-4c8f-b561-1ee418bc9e61
# ╟─82593cd0-1403-4597-8370-919c80494479
# ╟─6bf46802-0586-42f8-bd7e-9f0c5a36689b
# ╟─ae096ad2-3ae9-4440-a959-0d7d9a174f1d
# ╟─11557d6b-3a1e-416d-874f-b8d217976f76

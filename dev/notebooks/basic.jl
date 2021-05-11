### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 1ef174fa-16f0-11eb-328a-afc201effd2f
using Pkg, Printf

# ╔═╡ 55cfdab8-d792-11ea-271f-e7383e19997c
using PlutoUI;

# ╔═╡ 9e509f80-d485-11ea-0044-c5b7e750aacb
using NiLang

# ╔═╡ 37ed073a-d492-11ea-156f-1fb155128d0f
using Zygote, BenchmarkTools

# ╔═╡ 4d75f302-d492-11ea-31b9-bbbdb43f344e
using NiLang.AD

# ╔═╡ 627ea2fb-6530-4ea0-98ee-66be3db54411
html"""
<div align="center">
<a class="Header-link " href="https://github.com/GiggleLiu/NiLang.jl" data-hotkey="g d" aria-label="Homepage " data-ga-click="Header, go to dashboard, icon:logo">
  <svg class="octicon octicon-mark-github v-align-middle" height="32" viewBox="0 0 16 16" version="1.1" width="32" aria-hidden="true"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
</a>
<br>
<a href="https://raw.githubusercontent.com/GiggleLiu/NiLang.jl/master/notebooks/basic.jl" target="_blank" download>Download this notebook</a>
</div>
"""

# ╔═╡ 94b2b962-e02a-11ea-09a5-81b3226891ed
md"""# 连猩猩都能懂的可逆编程
### (Reversible programming made simple)
[https://github.com/JuliaReverse/NiLangTutorial/](https://github.com/JuliaReverse/NiLangTutorial/)

$(html"<br>")

**Jinguo Liu** (github: [GiggleLiu](https://github.com/GiggleLiu/))

*Postdoc, Institute of physics, Chinese academy of sciences* (when doing this project)

*Consultant, QuEra Computing* (current)

*Postdoc, Havard* (soon)
"""

# ╔═╡ a5ee60c8-e02a-11ea-3512-7f481e499f23
md"""
# Table of Contents
1. Reversible programming basics
2. Differentiate everything with a reversible programming language
4. Real world applications and benchmarks
"""

# ╔═╡ a11c4b60-d77d-11ea-1afe-1f2ab9621f42
md"""
## In this talk,
We use the reversible eDSL [NiLang](https://github.com/GiggleLiu/NiLang.jl) is a [Julia](https://julialang.org/) as our reversible programming tool.

A package that can differentiate everything.

![NiLang](https://raw.githubusercontent.com/GiggleLiu/NiLang.jl/master/docs/src/asset/logo3.png)

Authors:
[GiggleLiu](https://github.com/GiggleLiu), [Taine Zhao](https://github.com/thautwarm)
"""

# ╔═╡ e54a1be6-d485-11ea-0262-034c56e0fda8
md"""
## Sec I. Reversible programming basic

### Reversible function definition

A reversible function `f` is defined as
```julia
(~f)(f(x, y, z...)...) == (x, y, z...)
```
"""

# ╔═╡ d1628f08-ddfb-11ea-241a-c7e6c1a22212
md"""
##  Example 1: reversible adder
```math
\begin{align}
f &: x, y → x+y, y\\
{\small \mathrel{\sim}}f &: x, y → x-y, y
\end{align}
```
"""

# ╔═╡ 278ac6b6-e02c-11ea-1354-cd7ecd1099be
md"The reversible macro `@i` defines two functions, the function itself and its inverse."

# ╔═╡ a28d38be-d486-11ea-2c40-a377b74a05c1
@i function reversible_plus(x, y)
	x += y
end

# ╔═╡ e93f0bf6-d487-11ea-1baa-21d51ddb4a20
reversible_plus(2.0, 3.0)

# ╔═╡ fc932606-d487-11ea-303e-75ca8b7a02f6
(~reversible_plus)(5.0, 3.0)

# ╔═╡ e3d2b23a-ddfb-11ea-0f5e-e72ed299bb45
md"## The difference to a regular programming language"

# ╔═╡ a961e048-ddf2-11ea-0262-6d19eb82b36b
md"**Comment 1**: The return statement is not allowed, a reversible function returns input arguments directly."

# ╔═╡ 2d22f504-ddf1-11ea-28ec-5de6f4ee79bb
md"**Comment 2**: Every operation is reversible. `+=` is considered as reversible for integers and floating point numbers in NiLang, although for floating point numbers, there are *rounding errors*."

# ╔═╡ 7d08ac24-e143-11ea-2085-539fd9e35889
md"### A case where `+=` is not reversible"

# ╔═╡ 9fcdd77c-e0df-11ea-09e6-49a2861137e5
let
	x, y = 1e-20, 1e20
	x += y
	x -= y
	(x, y)
end

# ╔═╡ 0a1a8594-ddfc-11ea-119a-1997c86cd91b
md"""
## Use this function
"""

# ╔═╡ 0b4edb1a-ddf0-11ea-220c-91f2df7452e7
@i function reversible_plus2(x, y)
	reversible_plus(x, y)  # equivalent to `reversible_plus(x, y)`
	reversible_plus(x, y)
end

# ╔═╡ f875ecd6-ddef-11ea-22a1-619809d15b37
md"**Comment**: Inside a reversible function definition, a statement changes a variable *inplace*"

# ╔═╡ e7557bee-e0cc-11ea-1788-411e759b4766
reversible_plus2(2.0, 3.0)

# ╔═╡ cd7b2a2e-ddf5-11ea-04c4-f7583bbb5a53
md"A statement can be **uncalled** with `~`"

# ╔═╡ bc98a824-ddf5-11ea-1a6a-1f795452d3d0
@i function do_nothing(x, y)
	reversible_plus(x, y)
	~reversible_plus(x, y)  # uncall the expression
end

# ╔═╡ 05f8b91c-e0cd-11ea-09e3-f3c5c0e07e63
do_nothing(2.0, 3.0)

# ╔═╡ ac302844-e07b-11ea-35dd-e3e06054401b
md"## Example 2: Compute $x^5$"

# ╔═╡ b722e098-e07b-11ea-3483-01360fb6954e
@i function naive_power5(y, x::T) where T
	y = one(T)   # error 1: `=` is not reversible
	for i=1:5
		y *= x   # error 2: `*=` is not reversible
	end
end

# ╔═╡ bf8b722c-dfa4-11ea-196a-719802bc23c5
md"""
## Compute $x^5$ reversibly
"""

# ╔═╡ 330edc28-dfac-11ea-35a5-3144c4afbfcf
md"note: `*=` is not reversible for usual number systems"

# ╔═╡ 0a679e04-dfa7-11ea-0288-a1fa490c4387
@i function power5(x5, x4, x3, x2, x1, x)
	x1 += x
	x2 += x1 * x
	x3 += x2 * x
	x4 += x3 * x
	x5 += x4 * x
end

# ╔═╡ cc32cae8-dfab-11ea-0d0b-c70ea8de720a
power5(0.0, 0.0, 0.0, 0.0, 0.0, 2.0)

# ╔═╡ b4240c16-dfac-11ea-3a40-33c54436e3a3
md"# Don't make me so many input arguments!"

# ╔═╡ ade52358-dfac-11ea-2dd3-d3a691e7a8a2
@i function power5_twoinputs(x5, x::T) where T
	x1 ← zero(T)
	x2 ← zero(T)
	x3 ← zero(T)
	x4 ← zero(T)
	x1 += x
	x2 += x1 * x
	x3 += x2 * x
	x4 += x3 * x
	
	x5 += x4 * x
	
	x4 -= x3 * x
	x3 -= x2 * x
	x2 -= x1 * x
	x1 -= x
	x4 → zero(T)
	x3 → zero(T)
	x2 → zero(T)
	x1 → zero(T)
end

# ╔═╡ d86e2e5e-dfab-11ea-0053-6d52f1164bc5
power5_twoinputs(0.0, 2.0)

# ╔═╡ 7951b9ec-e030-11ea-32ee-b1de49378186
md"""
**Comment**:
`n ← zero(T)` is the variable allocation operation. It means
```
if n is defined
	error
else
	n = zero(T)
end
```
Its inverse is `n → zero(T)`. It means
```
@assert n == zero(T)
deallocate(n)
```
"""

# ╔═╡ 6bc97f5e-dfad-11ea-0c43-e30b6620e6e8
md"# Shorter: compute-copy-uncompute"

# ╔═╡ 80d24e9e-dfad-11ea-1dae-49568d534f10
@i function power5_twoinputs_shorter(x5, x::T) where T
	@routine begin  # compute
		@zeros T x1 x2 x3 x4
		x1 += x
		x2 += x1 * x
		x3 += x2 * x
		x4 += x3 * x
	end
	
	x5 += x4 * x   # copy
	
	~@routine    # uncompute
end

# ╔═╡ a8092b18-dfad-11ea-0989-474f37d05f73
power5_twoinputs_shorter(0.0, 2.0)

# ╔═╡ 43f0c2fc-e030-11ea-25d9-b323e6496a35
md"""**Comment**:
```
@routine statement
~@routine
```

is equivalent to
```
statement
~(statement)
```
This is the famous `compute-copy-uncompute` design pattern in reversible computing. Check this [reference](https://epubs.siam.org/doi/10.1137/0219046).
"""

# ╔═╡ b4ad5830-dfad-11ea-0057-055dda8cc9be
md"# How to compute x^1000?"

# ╔═╡ cf576d38-dfad-11ea-2682-7bd540db44a5
@i function power1000(x1000, x::T) where T
	@routine begin
		xs ← zeros(T, 1000)
		xs[1] += 1
		for i=2:1000
			xs[i] += xs[i-1] * x
		end
	end
	
	x1000 += xs[1000] * x
	
	~@routine
end

# ╔═╡ 35fff53c-dfae-11ea-3602-918a17d5a5fa
power1000(0.0, 1.001)

# ╔═╡ 9b9b5328-e030-11ea-1d00-f3341572734a
html"""
<h5>For loop</h5>
<div style="-webkit-column-count: 2; -moz-column-count: 2; column-count: 2; -webkit-column-rule: 1px dotted #e0e0e0; -moz-column-rule: 1px dotted #e0e0e0; column-rule: 1px dotted #e0e0e0; margin-top:30px">
<div style="display: inline-float">
	<center><strong>Forward</strong></center>
	<pre><code class="language-julia">
	for i=start:step:stop
		# do something
	end
	</code></pre>
</div>
<div style="display: inline-block;">
	<center><strong>Reverse</strong></center>
	<pre><code class="language-julia">
	for i=stop:-step:start
		# undo something
	end
	</code>
	</pre>
</div>
</div>
"""

# ╔═╡ f3b87892-e080-11ea-353d-8d81c52cf9ac
md"### Sometimes, a for loop can break down"

# ╔═╡ b27a3974-e030-11ea-0bcd-7f7035d55165
@i function power1000_bad(x1000, x::T) where T
	@routine begin
		xs ← zeros(T, 1000)
		xs[1] += 1
		for i=2:length(xs)
			xs[i] += xs[i-1] * x
			PUSH!(xs, @const zero(T))
		end
	end
	
	x1000 += xs[1000] * x
	
	~@routine
end

# ╔═╡ e5d47096-e030-11ea-1e87-5b9b1dbecfe0
power1000_bad(0.0, 1.001)

# ╔═╡ 9c62289a-dfae-11ea-0fe0-b1cb80a87704
md"#  Don't allocate for me!"

# ╔═╡ 88838bce-dfaf-11ea-1a72-7d15629cfcb0
md"""
Multipling two unsigned logarithmic numbers `x = exp(lx)` and `y = exp(ly)`
```
x * y = exp(lx) * exp(ly) = exp(lx + ly)
```
"""

# ╔═╡ a593f970-dfae-11ea-2d79-876030850dee
@i function power1000_noalloc(x1000, x::T) where T
	if x!= 0
		@routine begin
			absx ← zero(T)
			lx ← one(ULogarithmic{T})
			lx1000 ← one(ULogarithmic{T})
			absx += abs(x)
			lx *= convert(absx)
			for i=1:1000
				lx1000 *= lx
			end
		end
		x1000 += convert(lx1000)
		~@routine
	end
end

# ╔═╡ f448548e-dfaf-11ea-05c0-d5d177683445
power1000_noalloc(0.0, 1.001)

# ╔═╡ 65cd13ca-e031-11ea-3fc6-977792eb5f8c
html"""
<h5>If statement</h5>
<div style="-webkit-column-count: 2; -moz-column-count: 2; column-count: 2; -webkit-column-rule: 1px dotted #e0e0e0; -moz-column-rule: 1px dotted #e0e0e0; column-rule: 1px dotted #e0e0e0; margin-top:30px">
<div style="display: inline-float">
	<center><strong>Forward</strong></center>
	<pre><code class="language-julia">
	if (precondition, postcondition)
		# do A
	else
		# do B
	end
	</code></pre>
</div>
<div style="display: inline-block;">
	<center><strong>Reverse</strong></center>
	<pre><code class="language-julia">
	if (postcondition, precondition)
		# undo A
	else
		# undo B
	end
	</code>
	</pre>
</div>
</div>
"""

# ╔═╡ 53c02100-e08f-11ea-1f5d-8b2311b095d2
md"![](https://user-images.githubusercontent.com/6257240/116341762-78a31080-a7af-11eb-8376-d2ba0bf2b454.png)"

# ╔═╡ 75751b24-e0b8-11ea-2b37-9d138121345c
md"### You should not do"

# ╔═╡ 76b84de4-e031-11ea-0bcf-39b86a6b4552
@i function break_if(x)
	if x%2 == 1
		x += 1
	else
		x -= 1
	end
end

# ╔═╡ b1984d24-e031-11ea-3b13-3bd0119a2bcb
break_if(3)

# ╔═╡ 7f163d82-e0b8-11ea-2fe7-332bb4dee586
md"### You should do"

# ╔═╡ ddc6329e-e031-11ea-0e6e-e7332fa26e22
@i function happy_if(x)
	if (x%2 == 1, x%2 == 0)
		x += 1
	else
		x -= 1
	end
end

# ╔═╡ f3d5e1b0-e031-11ea-1a90-7bed88e28bad
happy_if(3)

# ╔═╡ ab67419a-dfae-11ea-27ba-09321303ad62
md"""# Wrap up

1. reversible arithmetic instructions `+=` and `-=`, besides, we have `SWAP`, `NEG`, `INC` and `ROT` et. al.
2. inverse statement `~`
3. there is no "`=`" operation in reversible computing, use "`←`" to allocate a new variable, and use "`→`" to deallocate an pre-emptied variable.
4. compute-uncompute macro `@routine` and `~@routine`
5. reversible control flow: `for` loop and `if` statement, the `while` statement is also available.
6. logarithmic number is reversible under `*=` and `/=`
"""

# ╔═╡ d5c2efbc-d779-11ea-11ad-1f5873b95628
md"""
![yeah](https://media1.tenor.com/images/40147f2eac14c0a7f18c34ecba73fa34/tenor.gif?itemid=7805520)
"""
# ![yeah](https://pic.chinesefontdesign.com/uploads/2017/03/chinesefontdesign.com_2017-03-07_08-19-24.gif)

# ╔═╡ 30af9642-e084-11ea-1f92-b52abfddcf06
md"# Sec II. Automatic differentiation in NiLang

### References
* Nextjournal [https://nextjournal.com/giggle/reverse-checkpointing](https://nextjournal.com/giggle/reverse-checkpointing)

* arXiv: 2003.04617
"

# ╔═╡ db1fab1c-e084-11ea-0bf0-b1fbe9e74b3f
html"""<h1><del>Auto</del>matic differentiation?</h1>"""

# ╔═╡ e1370f80-e0bc-11ea-2a90-d50cc762cbcb
md"When we start learning AD, we start by learning the backward rules of the matrix multiplication"

# ╔═╡ 3098411c-e0bc-11ea-2754-eb0afbd663de
function mymul!(out::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
	@assert size(A, 2) == size(B, 1) && size(out) == (size(A, 1), size(B, 2))
	for k=1:size(B, 2)
		for j=1:size(B, 1)
			for i=size(A, 1)
				@inbounds out[i, k] += A[i, j] * B[j, k]
			end
		end
	end
	return out
end

# ╔═╡ 3d0150ee-e0bd-11ea-0a5a-339465b496dc
md"Then, we learning how to use chain rule to chain different utilities."

# ╔═╡ 8016ff94-e0bc-11ea-3b9e-4f0676587edf
md"##### But wait! Why don't we start from the backward rules of `+` and `*`, then use the chain rule to derive the backward rule for matrix multiplication?"

# ╔═╡ 99108ace-e0bc-11ea-2744-d1b18db50ae1
md"# They are different"

# ╔═╡ b2337f26-e0bb-11ea-3da0-9507c35101ae
md"""
### Domain-specific autodiff (DS-AD)
* **Tensor**Flow
* **PyTorch**
* **Jax**
* **Flux (Zygote backended)**

### General Purposed autodiff (GP-AD)
* **Tapenade**
* **NiLang**
"""

# ╔═╡ 48db515c-e084-11ea-2eec-018b8545fa34
md"## Traditional AD uses checkpointing
Checkpoint every 100 steps. Blue and yellow objects are computing and re-computing. Here states 1 and state 101 are cached. Blue objects are computing, and yellow ones are re-computing. The state 100 is the desired state.
"

# ╔═╡ f531f556-e083-11ea-2f7e-77e110d6c53a
md"![](https://nextjournal.com/data/Qmes4v3ic2VrYQt6W9mWu4p6W53Gd1DmbDcYCuafbwTe7Y?filename=checkpointing.png&content-type=image/png)"

# ╔═╡ 62643fbc-e084-11ea-1b1f-39b87ff32b9e
md"## Reverse Computing
Reversible computing approach to free up memories (a) when no operations are reversible. (b) when all operations are reversible. Blue and yellow diamonds are reversible operations executed in forward and backward directions, red cubics are garbage variables.
"

# ╔═╡ 0bf54b08-e084-11ea-3d11-7be65f3ec022
md"![](https://nextjournal.com/data/QmPsgm4Z4mqVw2h2eC3RkGf96xTQp13KE9rdzmPeUe5KWN?filename=reversecomputing.png&content-type=image/png)"

# ╔═╡ 15f7c60a-e08e-11ea-31ea-a5cd055644db
md"## Difference Explained"

# ╔═╡ 55a3a260-d48e-11ea-06e2-1b7bd7bba6f5
md"""
![adprog](https://github.com/GiggleLiu/NiLang.jl/raw/master/docs/src/asset/adprog.png)
"""

# ╔═╡ 38014ad0-e08e-11ea-1905-198038ab7e5f
md"# Obtaining the gradient of norm in Zygote"

# ╔═╡ 2e6fe4da-d79d-11ea-1e90-f5215190395c
md"**Obtaining the gradient of the norm function**"

# ╔═╡ 6560c28c-e08e-11ea-1094-d333b88071ce
function regular_norm(x::AbstractArray{T}) where T
	res = zero(T)
	for i=1:length(x)
		@inbounds res += x[i]^2
	end
	return sqrt(res)
end

# ╔═╡ 744dd3c6-d492-11ea-0ed5-0fe02f99db1f
@benchmark Zygote.gradient($regular_norm, $(randn(1000))) seconds=1

# ╔═╡ f72246f8-e08e-11ea-3aa0-53f47a64f3e9
md"## The reversible counterpart"

# ╔═╡ f025e454-e08e-11ea-20d6-d139b9a6b301
@i function reversible_norm(res, y, x::AbstractArray{T}) where {T}
	for i=1:length(x)
		@inbounds y += x[i]^2
	end
	res += sqrt(y)
end

# ╔═╡ 8fedd65a-e08e-11ea-27f4-03bf9ed65875
let x = randn(1000)
	@assert Zygote.gradient(regular_norm, x)[1] ≈ NiLang.AD.gradient(reversible_norm, (0.0, 0.0, x), iloss=1)[3]
end

# ╔═╡ 8ad60dc0-d492-11ea-2cb3-1750b39ddf86
@benchmark NiLang.AD.gradient($reversible_norm, (0.0, 0.0, $(randn(1000))), iloss=1)

# ╔═╡ 7bab4614-d77e-11ea-037c-8d1f432fc3b8
md"""
![yeah](https://media1.tenor.com/images/40147f2eac14c0a7f18c34ecba73fa34/tenor.gif?itemid=7805520)
"""
# ![yeah](https://pic.chinesefontdesign.com/uploads/2017/03/chinesefontdesign.com_2017-03-07_08-19-24.gif)


# ╔═╡ fcca27ba-d4a4-11ea-213a-c3e2305869f1
#**1. The bundle adjustment jacobian benchmark**
#$(LocalResource("ba-origin.png"))
#![ba](https://github.com/JuliaReverse/NiBundleAdjustment.jl/raw/master/benchmarks/adbench.png)

#**2. The Gaussian mixture model benchmark**
#$(LocalResource("gmm-origin.png"))
#![gmm](https://github.com/JuliaReverse/NiGaussianMixture.jl/raw/master/benchmarks/adbench.png)

md"""
# Sec III. Applications in real world and benchmarks
"""

# ╔═╡ 519dc834-e092-11ea-2151-57ef23810b84
md"""
## 1. Bundle Adjustment (Jacobian)
![bundle adjustment](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRgpGSCWRjHDDaIYQX5ejhMvyKY_GFhynVoQg&usqp=CAU)

*Srajer, Filip, Zuzana Kukelova, and Andrew Fitzgibbon. "A benchmark of selected algorithmic differentiation tools on some problems in computer vision and machine learning." Optimization Methods and Software 33.4-6 (2018): 889-906.*

### Benchmarks
**Devices**
* CPU: Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz
* GPU: Nvidia Titan V. 

**Github Repos** 
* [https://github.com/microsoft/ADBench](https://github.com/microsoft/ADBench)
* [https://github.com/JuliaReverse/NiBundleAdjustment.jl](https://github.com/JuliaReverse/NiBundleAdjustment.jl)
"""

# ╔═╡ c89108f0-e092-11ea-0fe2-efad85008b28
html"""
<div style="float: left"><img src="https://adbenchresults.blob.core.windows.net/plots/2020-03-29_15-46-08_70e2e936bea81eebf0de78ce18d4d196daf1204e/static/jacobian/BA%20[Jacobian]%20-%20Release%20Graph.png" width=500/></div>
"""

# ╔═╡ 2ec4c700-e093-11ea-06ff-47d2c21a068f
md"""##### NiLang.AD and Tapenade
![](https://user-images.githubusercontent.com/6257240/116341804-907a9480-a7af-11eb-934f-7eb94803f5f2.png)"""

# ╔═╡ 474aa228-e092-11ea-042b-bdfaeb99f16f
md"""
## 2. Gaussian Mixture Model (Gradient)
![gmm](https://prateekvjoshi.files.wordpress.com/2013/06/multimodal.jpg)
"""

# ╔═╡ 2baaff10-d56c-11ea-2a23-bfa3a7ae2e4b
md"""
### Benchmarks
*Srajer, Filip, Zuzana Kukelova, and Andrew Fitzgibbon. "A benchmark of selected algorithmic differentiation tools on some problems in computer vision and machine learning." Optimization Methods and Software 33.4-6 (2018): 889-906.*

**Devices**
* CPU: Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz

**Github Repos** 
* [https://github.com/microsoft/ADBench](https://github.com/microsoft/ADBench)
* [https://github.com/JuliaReverse/NiGaussianMixture.jl](https://github.com/JuliaReverse/NiGaussianMixture.jl)
"""

# ╔═╡ 102fbf2e-d56b-11ea-189d-c78d56c0a924
html"""
<h5>Results from the original benchmark<h5>
<img src="https://adbenchresults.blob.core.windows.net/plots/2020-03-29_15-46-08_70e2e936bea81eebf0de78ce18d4d196daf1204e/static/jacobian/GMM%20(10k)%20[Jacobian]%20-%20Release%20Graph.png" width=5000/>
"""

# ╔═╡ cc0d5622-d788-11ea-19cd-3bf6864d9263
md"""##### Including NiLang.AD
![](https://github.com/JuliaReverse/NiLangTutorial/blob/master/notebooks/asset/benchmarks_gmm.png?raw=true)"""

# ╔═╡ a1646ef0-e091-11ea-00f1-e7c246e191ff
md"## 3. Solve the memory wall problem in machine learning"

# ╔═╡ b18b3ae8-e091-11ea-24a1-e968b70b217c
html"""
Learning a ring distribution with NICE network, before and after training

<img style="float:left" src="https://giggleliu.github.io/NiLang.jl/dev/asset/nice_before.png" width=340/>
<img src="https://giggleliu.github.io/NiLang.jl/dev/asset/nice_after.png" width=340/>

<h5>References</h5>
<ul>
<li><a href="https://arxiv.org/abs/1410.8516">arXiv: 1410.8516</li>
<li><a href="https://giggleliu.github.io/NiLang.jl/dev/examples/nice/#NICE-network-1">NiLang's documentation</a></li>
</ul>
"""

# ╔═╡ bf3774de-e091-11ea-3372-ef56452158e6
md"""
## 4. Solve the spinglass ground state configuration
Obtaining the optimal configuration of a spinglass problem on a $28 \times 28$ square lattice.

![](https://user-images.githubusercontent.com/6257240/116342088-067efb80-a7b0-11eb-935e-0b5e29010a22.png)

##### References
Jin-Guo Liu, Lei Wang, Pan Zhang, **arXiv 2008.06888**
"""

# ╔═╡ c8e4f7a6-e091-11ea-24a3-4399635a41a5
md"""
## 5. Optimizing problems in finance
Gradient based optimization of Sharpe rate.


600x acceleration comparing with using pure Zygote.

##### References
* Han Li's Github repo: [https://github.com/HanLi123/NiLang](https://github.com/HanLi123/NiLang) and his Zhihu blog [猴子掷骰子](https://zhuanlan.zhihu.com/c_1092471228488634368).
"""

# ╔═╡ bc872296-e09f-11ea-143b-9bfd5e52b14f
md"""## 6. Accelerate the performance critical part of variational mean field

[https://github.com/quantumlang/NiLangTest/pull/1](https://github.com/quantumlang/NiLangTest/pull/1)

600x acceleration comparing with using pure Zygote.
"""

# ╔═╡ e7b21fce-e091-11ea-180c-7b42e00598a9
md"""# Thank you!
Special thanks to my collaborator **Taine Zhao** and (ex-)advisor **Lei Wang**.

QuEra computing (a quantum computing company located in Boston) is hiring people.
"""

# ╔═╡ 7c79975c-d789-11ea-30b1-67ff05418cdb
md"""
![yeah](https://media1.tenor.com/images/40147f2eac14c0a7f18c34ecba73fa34/tenor.gif?itemid=7805520)
"""
# ![yeah](https://pic.chinesefontdesign.com/uploads/2017/03/chinesefontdesign.com_2017-03-07_08-19-24.gif)

# ╔═╡ 5f1c3f6c-d48b-11ea-3eb0-357fd3ece4fc
md"""
## Sec IV. More about number systems

 
* Integers are reversible under (`+=`, `-=`).
* Floating point number system is **irreversible** under (`+=`, `-=`) and (`*=`, `/=`).
* [Fixedpoint number system](https://github.com/JuliaMath/FixedPointNumbers.jl) are reversible under (`+=`, `-=`)
* [Logarithmic number system](https://github.com/cjdoris/LogarithmicNumbers.jl) is reversible under (`*=`, `/=`)
"""

# ╔═╡ 11ddebfe-d488-11ea-223a-e9403f6ec8de
md"""
##### Example 1: Affine transformation with rounding error

```julia
y = A * x + b
```
"""

# ╔═╡ 030e592e-d488-11ea-060d-97a3bb6353b7
@i function reversible_affine!(y!::AbstractVector{T}, W::AbstractMatrix{T}, b::AbstractVector{T}, x::AbstractVector{T}) where T
    @safe @assert size(W) == (length(y!), length(x)) && length(b) == length(y!)
    for j=1:size(W, 2)
        for i=1:size(W, 1)
            @inbounds y![i] += W[i,j]*x[j]
        end
    end
    for i=1:size(W, 1)
        @inbounds y![i] += b[i]
    end
end

# ╔═╡ c8d26856-d48a-11ea-3cd3-1124cd172f3a
begin
	W = randn(10, 10)
	b = randn(10)
	x = randn(10)
end;

# ╔═╡ 37c4394e-d489-11ea-174c-b13bdddbe741
yout, Wout, bout, xout = reversible_affine!(zeros(10), W, b, x)

# ╔═╡ fef54688-d48a-11ea-340b-295b88d21382
# should be restored to 0, but not!
yin, Win, bin, xin = (~reversible_affine!)(yout, Wout, bout, xout)

# ╔═╡ 259a2852-d48c-11ea-0f01-b9634850e09d
md"""
### Reversible arithmetic functions

Computing basic functions like `power`, `exp` and `besselj` is not trivial for reversible programming.
There is no efficient constant memory algorithm using pure fixed point numbers only.
"""

# ╔═╡ f06fb004-d79f-11ea-0d60-8151019bf8c7
md"""
##### Example 2: Computing power function
To compute `x ^ n` reversiblly with fixed point numbers,
we need to either allocate a vector of size $O(n)$ or suffer from polynomial time overhead. It does not show the advantage to checkpointing.
"""

# ╔═╡ 26a8a42c-d7a1-11ea-24a3-45bc6e0674ea
@i function i_power_cache(y!::T, x::T, n::Int) where T
    @routine @invcheckoff begin
        cache ← zeros(T, n)  # allocate a buffer of size n
		cache[1] += x
        for i=2:n
            cache[i] += cache[i-1] * x
        end
    end

    y! += cache[n]

    ~@routine  # uncompute cache
end

# ╔═╡ 399552c4-d7a1-11ea-36bb-ad5ca42043cb
# To check the function
i_power_cache(Fixed43(0.0), Fixed43(0.99), 100)

# ╔═╡ 4bb19760-d7bf-11ea-12ed-4d9e4efb3482
md"""
##### Example 3: reversible thinker, the logarithmic number approach

With **logarithmic numbers**, we can still utilize reversibility. Fixed point numbers and logarithmic numbers can be converted via "a fast binary logarithm algorithm".

##### References
* [1] C. S. Turner, "A Fast Binary Logarithm Algorithm", IEEE Signal Processing Mag., pp. 124,140, Sep. 2010.
"""

# ╔═╡ 5a8ba8f4-d493-11ea-1839-8ba81f86799d
@i function i_power_lognumber(y::T, x::T, n::Int) where T
    @routine @invcheckoff begin
        lx ← one(ULogarithmic{T})
        ly ← one(ULogarithmic{T})
        ## convert `x` to a logarithmic number
        ## Here, `*=` is reversible for log numbers
        lx *= convert(x)
        for i=1:n
            ly *= lx
        end
    end

    ## convert back to fixed point numbers
    y += convert(ly)

    ~@routine
end

# ╔═╡ a625a922-d493-11ea-1fe9-bdd4a694cde0
# To check the function
i_power_lognumber(Fixed43(0.0), Fixed43(0.99), 100)

# ╔═╡ 4fd20ed2-d7a2-11ea-206e-13799234913f
md"**Less allocation, better speed**"

# ╔═╡ 692dfb44-d7a1-11ea-00da-af6550bc0622
@benchmark i_power_cache(Fixed43(0.0), Fixed43(0.99), 100)

# ╔═╡ 7e4ee09c-d7a1-11ea-0e56-c1921012bc30
@benchmark i_power_lognumber(Fixed43(0.0), Fixed43(0.99), 100)

# ╔═╡ 4c209bbe-d7b1-11ea-0628-33eb8d664f5b
md"""##### Example 4: The first kind Bessel function computed with Taylor expansion
```math
J_\nu(z) = \sum\limits_{n=0}^{\infty} \frac{(z/2)^\nu}{\Gamma(k+1)\Gamma(k+\nu+1)} (-z^2/4)^{n}
```


"""

# ╔═╡ fd44a3d4-d7a4-11ea-24ea-09456ff2c53d
@i function ibesselj(y!::T, ν, z::T; atol=1e-8) where T
	if z == 0
		if ν == 0
			out! += 1
		end
	else
		@routine @invcheckoff begin
			k ← 0
			@ones ULogarithmic{T} lz halfz halfz_power_2 s
			@zeros T out_anc
			lz *= convert(z)
			halfz *= lz / 2
			halfz_power_2 *= halfz ^ 2
			# s *= (z/2)^ν/ factorial(ν)
			s *= halfz ^ ν
			for i=1:ν
				s /= i
			end
			out_anc += convert(s)
			while (s.log > -25, k!=0) # upto precision e^-25
				k += 1
				# s *= 1 / k / (k+ν) * (z/2)^2
				@routine begin
					@zeros Int kkv kv
					kv += k+ ν
					kkv += kv*k
				end
				s *= halfz_power_2 / kkv
				if k%2 == 0
					out_anc += convert(s)
				else
					out_anc -= convert(s)
				end
				~@routine
			end
		end
		y! += out_anc
		~@routine
	end
end

# ╔═╡ 84272664-d7b7-11ea-2e37-dffd2023d8d6
md"z = $(@bind z Slider(0:0.01:10; default=1.0))"

# ╔═╡ 900e2ea4-d7b8-11ea-3511-6f12d95e638a
begin
	y = ibesselj(Fixed43(0.0), 2, Fixed43(z))[1]
	gz = NiLang.AD.gradient(Val(1), ibesselj, (Fixed43(0.0), 2, Fixed43(z)))[3]
end;

# ╔═╡ d76be888-d7b4-11ea-2989-2174682ead76
let
	md"""
| ``z`` | ``y`` | ``\partial y/\partial z`` |
| ----  | ----- | -------- |
| $(@sprintf "%.5f" z) | $(@sprintf "%.5f" y) | $(@sprintf "%.5f" gz) |
"""
end

# ╔═╡ 85c9edcc-d789-11ea-14c8-71697cd6a047
md"""
![yeah](https://media1.tenor.com/images/40147f2eac14c0a7f18c34ecba73fa34/tenor.gif?itemid=7805520)
"""
# ![yeah](https://pic.chinesefontdesign.com/uploads/2017/03/chinesefontdesign.com_2017-03-07_08-19-24.gif)

# ╔═╡ Cell order:
# ╟─1ef174fa-16f0-11eb-328a-afc201effd2f
# ╟─627ea2fb-6530-4ea0-98ee-66be3db54411
# ╟─94b2b962-e02a-11ea-09a5-81b3226891ed
# ╟─a5ee60c8-e02a-11ea-3512-7f481e499f23
# ╟─a11c4b60-d77d-11ea-1afe-1f2ab9621f42
# ╟─e54a1be6-d485-11ea-0262-034c56e0fda8
# ╟─55cfdab8-d792-11ea-271f-e7383e19997c
# ╟─d1628f08-ddfb-11ea-241a-c7e6c1a22212
# ╠═9e509f80-d485-11ea-0044-c5b7e750aacb
# ╟─278ac6b6-e02c-11ea-1354-cd7ecd1099be
# ╠═a28d38be-d486-11ea-2c40-a377b74a05c1
# ╠═e93f0bf6-d487-11ea-1baa-21d51ddb4a20
# ╠═fc932606-d487-11ea-303e-75ca8b7a02f6
# ╟─e3d2b23a-ddfb-11ea-0f5e-e72ed299bb45
# ╟─a961e048-ddf2-11ea-0262-6d19eb82b36b
# ╟─2d22f504-ddf1-11ea-28ec-5de6f4ee79bb
# ╟─7d08ac24-e143-11ea-2085-539fd9e35889
# ╠═9fcdd77c-e0df-11ea-09e6-49a2861137e5
# ╟─0a1a8594-ddfc-11ea-119a-1997c86cd91b
# ╠═0b4edb1a-ddf0-11ea-220c-91f2df7452e7
# ╟─f875ecd6-ddef-11ea-22a1-619809d15b37
# ╠═e7557bee-e0cc-11ea-1788-411e759b4766
# ╟─cd7b2a2e-ddf5-11ea-04c4-f7583bbb5a53
# ╠═bc98a824-ddf5-11ea-1a6a-1f795452d3d0
# ╠═05f8b91c-e0cd-11ea-09e3-f3c5c0e07e63
# ╟─ac302844-e07b-11ea-35dd-e3e06054401b
# ╠═b722e098-e07b-11ea-3483-01360fb6954e
# ╟─bf8b722c-dfa4-11ea-196a-719802bc23c5
# ╟─330edc28-dfac-11ea-35a5-3144c4afbfcf
# ╠═0a679e04-dfa7-11ea-0288-a1fa490c4387
# ╠═cc32cae8-dfab-11ea-0d0b-c70ea8de720a
# ╟─b4240c16-dfac-11ea-3a40-33c54436e3a3
# ╠═ade52358-dfac-11ea-2dd3-d3a691e7a8a2
# ╠═d86e2e5e-dfab-11ea-0053-6d52f1164bc5
# ╟─7951b9ec-e030-11ea-32ee-b1de49378186
# ╟─6bc97f5e-dfad-11ea-0c43-e30b6620e6e8
# ╠═80d24e9e-dfad-11ea-1dae-49568d534f10
# ╠═a8092b18-dfad-11ea-0989-474f37d05f73
# ╟─43f0c2fc-e030-11ea-25d9-b323e6496a35
# ╟─b4ad5830-dfad-11ea-0057-055dda8cc9be
# ╠═cf576d38-dfad-11ea-2682-7bd540db44a5
# ╠═35fff53c-dfae-11ea-3602-918a17d5a5fa
# ╟─9b9b5328-e030-11ea-1d00-f3341572734a
# ╟─f3b87892-e080-11ea-353d-8d81c52cf9ac
# ╠═b27a3974-e030-11ea-0bcd-7f7035d55165
# ╠═e5d47096-e030-11ea-1e87-5b9b1dbecfe0
# ╟─9c62289a-dfae-11ea-0fe0-b1cb80a87704
# ╟─88838bce-dfaf-11ea-1a72-7d15629cfcb0
# ╠═a593f970-dfae-11ea-2d79-876030850dee
# ╠═f448548e-dfaf-11ea-05c0-d5d177683445
# ╟─65cd13ca-e031-11ea-3fc6-977792eb5f8c
# ╟─53c02100-e08f-11ea-1f5d-8b2311b095d2
# ╟─75751b24-e0b8-11ea-2b37-9d138121345c
# ╠═76b84de4-e031-11ea-0bcf-39b86a6b4552
# ╠═b1984d24-e031-11ea-3b13-3bd0119a2bcb
# ╟─7f163d82-e0b8-11ea-2fe7-332bb4dee586
# ╠═ddc6329e-e031-11ea-0e6e-e7332fa26e22
# ╠═f3d5e1b0-e031-11ea-1a90-7bed88e28bad
# ╟─ab67419a-dfae-11ea-27ba-09321303ad62
# ╟─d5c2efbc-d779-11ea-11ad-1f5873b95628
# ╟─30af9642-e084-11ea-1f92-b52abfddcf06
# ╟─db1fab1c-e084-11ea-0bf0-b1fbe9e74b3f
# ╟─e1370f80-e0bc-11ea-2a90-d50cc762cbcb
# ╠═3098411c-e0bc-11ea-2754-eb0afbd663de
# ╟─3d0150ee-e0bd-11ea-0a5a-339465b496dc
# ╟─8016ff94-e0bc-11ea-3b9e-4f0676587edf
# ╟─99108ace-e0bc-11ea-2744-d1b18db50ae1
# ╟─b2337f26-e0bb-11ea-3da0-9507c35101ae
# ╟─48db515c-e084-11ea-2eec-018b8545fa34
# ╟─f531f556-e083-11ea-2f7e-77e110d6c53a
# ╟─62643fbc-e084-11ea-1b1f-39b87ff32b9e
# ╟─0bf54b08-e084-11ea-3d11-7be65f3ec022
# ╟─15f7c60a-e08e-11ea-31ea-a5cd055644db
# ╟─55a3a260-d48e-11ea-06e2-1b7bd7bba6f5
# ╟─38014ad0-e08e-11ea-1905-198038ab7e5f
# ╟─2e6fe4da-d79d-11ea-1e90-f5215190395c
# ╠═6560c28c-e08e-11ea-1094-d333b88071ce
# ╠═37ed073a-d492-11ea-156f-1fb155128d0f
# ╠═744dd3c6-d492-11ea-0ed5-0fe02f99db1f
# ╟─f72246f8-e08e-11ea-3aa0-53f47a64f3e9
# ╠═f025e454-e08e-11ea-20d6-d139b9a6b301
# ╠═4d75f302-d492-11ea-31b9-bbbdb43f344e
# ╠═8fedd65a-e08e-11ea-27f4-03bf9ed65875
# ╠═8ad60dc0-d492-11ea-2cb3-1750b39ddf86
# ╟─7bab4614-d77e-11ea-037c-8d1f432fc3b8
# ╟─fcca27ba-d4a4-11ea-213a-c3e2305869f1
# ╟─519dc834-e092-11ea-2151-57ef23810b84
# ╟─c89108f0-e092-11ea-0fe2-efad85008b28
# ╟─2ec4c700-e093-11ea-06ff-47d2c21a068f
# ╟─474aa228-e092-11ea-042b-bdfaeb99f16f
# ╟─2baaff10-d56c-11ea-2a23-bfa3a7ae2e4b
# ╟─102fbf2e-d56b-11ea-189d-c78d56c0a924
# ╟─cc0d5622-d788-11ea-19cd-3bf6864d9263
# ╟─a1646ef0-e091-11ea-00f1-e7c246e191ff
# ╟─b18b3ae8-e091-11ea-24a1-e968b70b217c
# ╟─bf3774de-e091-11ea-3372-ef56452158e6
# ╟─c8e4f7a6-e091-11ea-24a3-4399635a41a5
# ╟─bc872296-e09f-11ea-143b-9bfd5e52b14f
# ╟─e7b21fce-e091-11ea-180c-7b42e00598a9
# ╟─7c79975c-d789-11ea-30b1-67ff05418cdb
# ╟─5f1c3f6c-d48b-11ea-3eb0-357fd3ece4fc
# ╟─11ddebfe-d488-11ea-223a-e9403f6ec8de
# ╠═030e592e-d488-11ea-060d-97a3bb6353b7
# ╠═c8d26856-d48a-11ea-3cd3-1124cd172f3a
# ╠═37c4394e-d489-11ea-174c-b13bdddbe741
# ╠═fef54688-d48a-11ea-340b-295b88d21382
# ╟─259a2852-d48c-11ea-0f01-b9634850e09d
# ╟─f06fb004-d79f-11ea-0d60-8151019bf8c7
# ╠═26a8a42c-d7a1-11ea-24a3-45bc6e0674ea
# ╠═399552c4-d7a1-11ea-36bb-ad5ca42043cb
# ╟─4bb19760-d7bf-11ea-12ed-4d9e4efb3482
# ╠═5a8ba8f4-d493-11ea-1839-8ba81f86799d
# ╠═a625a922-d493-11ea-1fe9-bdd4a694cde0
# ╟─4fd20ed2-d7a2-11ea-206e-13799234913f
# ╠═692dfb44-d7a1-11ea-00da-af6550bc0622
# ╠═7e4ee09c-d7a1-11ea-0e56-c1921012bc30
# ╟─4c209bbe-d7b1-11ea-0628-33eb8d664f5b
# ╠═fd44a3d4-d7a4-11ea-24ea-09456ff2c53d
# ╟─84272664-d7b7-11ea-2e37-dffd2023d8d6
# ╠═900e2ea4-d7b8-11ea-3511-6f12d95e638a
# ╟─d76be888-d7b4-11ea-2989-2174682ead76
# ╟─85c9edcc-d789-11ea-14c8-71697cd6a047

### A Pluto.jl notebook ###
# v0.15.0

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

# ╔═╡ f3e235e7-76b9-4c39-bc70-038539838ff4
begin
	using Revise, Viznet, Compose, PlutoUI, Random, TikzPictures
	function leftright(a, b; width=600, leftcellwidth=0.5)
		HTML("""
<style>
table.nohover tr:hover td {
   background-color: white !important;
}</style>
			
<table width=$(width)px class="nohover" style="border:none">
<tr>
	<td width=$(leftcellwidth*width)>$(html(a))</td>
	<td width=$((1-leftcellwidth)*width)>$(html(b))</td>
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

# ╔═╡ 3b0fd2b5-5c6d-4d56-9e48-cda1493b4c72
using NiLang

# ╔═╡ a7810352-7967-460d-abd7-361a324c20a9
using ForwardDiff: Dual

# ╔═╡ a56445b5-e530-4035-9ac6-a2d196a6276a
using NiLang.AD: GVar

# ╔═╡ c6bd40af-50ed-4cee-8043-60b2bac05058
using NiLang: bennett

# ╔═╡ 873ef2c2-653e-425e-9732-b1ed19f7a0b7
using TreeverseAlgorithm

# ╔═╡ 141e21c0-1bdf-4e6b-b76d-129567a1180f
using LinearAlgebra

# ╔═╡ ac35b26c-0585-4d2a-8fbf-bda9b141d6af
using Test

# ╔═╡ d4726239-81af-4792-8472-c680508449c6
using BenchmarkTools

# ╔═╡ e20e2d2e-4b28-4e32-8d80-ce029928a094
html"""
<script>
document.body.onkeyup = function(e) {
if (e.ctrlKey && e.altKey && e.which == 80) {
    present();
} else if (e.ctrlKey && e.which == 37) {
	var prev_button = document.querySelector(".changeslide.prev");
	prev_button.dispatchEvent(new Event('click'));
} else if (e.ctrlKey && e.which == 39) {
	var prev_button = document.querySelector(".changeslide.next");
	prev_button.dispatchEvent(new Event('click'));
  }
};
document.body.onclick = function(e) {
	if (e.target.tagName == 'BODY'){
		e.preventDefault();
		var prev_button = document.querySelector(".changeslide.next");
		prev_button.dispatchEvent(new Event('click'));
} else if (e.target.tagName == 'PLUTO-SHOULDER'){
	e.preventDefault();
	var prev_button = document.querySelector(".changeslide.prev");
	prev_button.dispatchEvent(new Event('click'));
	}
};
</script>

<style>
mjx-assistive-mml { display: none !important; }
</style>
"""

# ╔═╡ 8308df59-3faa-4abf-8f05-119bbae48f64
let
	github = html"""<a class="Header-link " href="https://github.com/GiggleLiu/NiLang.jl" data-hotkey="g d" aria-label="Homepage " data-ga-click="Header, go to dashboard, icon:logo">
  <img src="https://avatars.githubusercontent.com/u/6257240?v=4" width=25> GiggleLiu
</a>"""
	md"# Pebble games - Time and space to differentiate a program

-- Jinguo Liu ($github)
"
end

# ╔═╡ a3532a83-9fd3-4d24-b1bb-b52457317e51
html"""A postdoc in Mikhail Lukin's group, department of physics <br><br>
<img src="https://1000logos.net/wp-content/uploads/2017/02/Harvard-Logo.png" width=55> Harvard university<br><br>
<img src="https://static1.squarespace.com/static/5dcebcb378a43f6976d84698/t/5dcf099a5f767722ba4b9cdd/1605640592777/?format=1500w" width=80/> Quera Computing
"""

# ╔═╡ 15657e4b-848e-43ad-a99f-37143d11705e
md"# Table of contents
1. An introduction to reversible computing
2. A reversible eDSL NiLang
3. Automatic differentiating a reversible computing language
"

# ╔═╡ 34a6b7f4-7d72-485d-86dc-f4b1ba6174eb
md"## Let's start from physics"

# ╔═╡ 67d1b500-964e-4668-a7d0-ed93886446ca
let
	img1 = html"""
<img src="http://cen.acs.org/content/dam/cen/98/5/WEB/09805-feature3-ski1.jpg" height=210/>
"""
	img2 = html"""
<img src="https://www.ux1.eiu.edu/~cfadd/1360/29MagFlds/Images/Fig29.26.jpg" height=210/>"""
	leftright(updown(img2, html"<div align='center'>electromagnetic force</div>"), updown(img1, html"<div align='center'>friction</div>"))
end

# ╔═╡ 673992ed-6963-400a-a69b-d65d26c4f443
md"Both can be explained by reversible quantum dynamics."

# ╔═╡ 6078758e-b392-4bdb-a1e7-44b135ce900e
md"""
## How come our programming style is irreversible?
"""

# ╔═╡ 41e06e2a-b482-4e0f-8569-fee2ffd8aaaf
function find_maximum(x::AbstractVector)
	@assert !isempty(x)   # error handling
	m = x[1]              # assignment (a)
	for i=2:length(x)
		m = max(m, x[i])  # assignment (b)
	end
	return m              # function return
end

# ╔═╡ dcf53d46-e259-4101-8530-9621094ee586
TikzPicture(L"""
\draw [black, thick,->] (0, 0) -- (1, 0);
\draw [black, thick,->,dashed] (1, 0) .. controls (1.5, 0.5) .. (2, 0);
\node at (1.5, 0.5) {goto $\ldots$};
\draw [black, thick,->] (2, 0) -- (3, 0);
       
\def\x{4};
\draw [black, thick,<-] (\x, 0) -- (\x+1, 0);
\draw [black, thick,<-,dashed] (\x+1, 0) .. controls (\x+1.5, 0.5) .. (\x+2, 0);
\draw [black, thick,<-,dashed] (\x+1, 0) -- (\x+2, 0);
\node at (\x+1.5, 0.5) {comefrom?};
\draw [black, thick,<-] (\x+2, 0) -- (\x+3, 0);
       
\node at (1.5, -0.2) {call};
\node at (\x+1.5, -0.2) {uncall};
""", options="scale=2.0", preamble="")

# ╔═╡ 6a88d26c-c895-4852-ab4f-37297b848731
md"""
* **Information**: the uncertainty, quantified of *information entropy*.
* **Information erasure**: make the system more certain, e.g.
```julia
m = max(m, x[i])
```
quantified by the decrease of information entropy.
"""

# ╔═╡ f9675365-36aa-430c-b747-3bc4f602e6fb
md"## Information erasure requires dissipating heat to the environment!"

# ╔═╡ 46eb4ba9-dce6-4711-9c4d-3f16de6240de
leftright(html"""
<img src="https://images-na.ssl-images-amazon.com/images/I/51o-kZ4x6fL._SX351_BO1,204,203,200_.jpg" width=200/>""", md"Feynman, Richard P.

**Feynman Lectures on Computation**
	
(2018)")

# ╔═╡ 046f7559-4af9-4982-b5c3-335add0911d7
html"""
<div align=center><img src="https://user-images.githubusercontent.com/6257240/122632611-ef6ad480-d0a1-11eb-976c-a3e7c5dfdb9a.png" width=500/></div>
"""

# ╔═╡ 0a039bfa-571e-4fad-b73c-1324d08777fc
html"""
<div align=center><img src="https://user-images.githubusercontent.com/6257240/122682827-b168d000-d1c9-11eb-930c-0ff13a2bf631.png" width=300/></div>
"""

# ╔═╡ 3f1e4d7a-32a7-4c7e-92dd-465bac925e63
md"""
Compress the boxes from size $V$ to size $V/2$, the process is isothermal.
"""

# ╔═╡ f68bcfb6-97ce-48d1-b0b8-e8466d4ac879
md"""
Case 1: we know nothing about the system. The gas does work
```math
\begin{align}
&pV = N k T\\
&W_{\rm gas} = \int_{V}^{V/2} p dV = -NkT\log 2
\end{align}
```
"""

# ╔═╡ 3d4ba750-8d62-48ac-bf96-691397689ddc
md"""
Case 2: we know one bit knowledge about each box:
* 1: the atom is in the right half
* 0: the atom is in the left half

```math
W_{\rm gas} = 0
```
"""

# ╔═╡ f4cb9212-181f-4338-b858-1d99c7f415e9
md"Erasing each bit information comes along with $kT \log 2$ heat dissipation!!"

# ╔═╡ 31bde262-6352-4be0-b5cc-1781e3df2268
md"Later people proved it from the microscopic picture. [Reeb, 2014]"

# ╔═╡ 83ff3fc3-bcd8-4235-a42f-1d75c7d6aa5b
md"## Computing architectures"

# ╔═╡ b308e270-6b40-4946-ac92-c705823f2c1e
let
	txt1 = md"Traditional irreversible computer
	
$E \sim 10^8 kT$"
	img1 = html"""<img src="https://www.computerhope.com/jargon/c/computer-laptop-2in1.jpg" width=120/>"""
	txt2 = md"DNA copying is a living copy machine

$E \sim 100k T$"
	img2 = html"""
<img src="https://s3-us-west-2.amazonaws.com/courses-images/wp-content/uploads/sites/110/2016/06/02172248/DNA_replication_split_horizontal.svg_-1024x508.png" width=300/>
"""
	txt3 = md"""
Adiabatic CMOS [Athas, 1994]

$E \sim 10^6 kT$
"""
	img3 = html"""
<img src="https://user-images.githubusercontent.com/6257240/122668453-287c7500-d186-11eb-962f-cc478be1dafe.png" width=350 style="margin-bottom:25px"/>
"""
	txt4 = md"""Adiabatic superconducting devices [Takeuchi, 2014]

$E \sim kT$
"""
	img4 = html"""
<img src="https://scitechdaily.com/images/Magnet-Levitates-Above-Superconductor.jpg" width=300/>
"""
	updown(leftright(updown(img1, txt1), updown(img2, txt2)), leftright(updown(img3, txt3), updown(img4, txt4)))
end

# ╔═╡ e483b3d4-d01c-4a98-8e68-e8120a7d95a7
md"# Summary
* An isolated system is reversible,
* Our programs are not reversible,
    * Need a heatbath
    * Dissipate heat to heat bath: ``kT \log 2``/bit (Landauer's principle),


![](https://user-images.githubusercontent.com/6257240/123520518-22ebc700-d67f-11eb-8af1-a452605cc1d8.png)

*Youtube*: Michael P. Frank: Fundamental Physics of Reversible Computing — An Introduction, Part 1

*Loophole*: need to take algorithmic overheads into consideration!
"

# ╔═╡ 20c34526-c7c4-11eb-21fa-d706fd684a4c
md"# A short introduction to the reversible programming"

# ╔═╡ 3f96abdf-fb5f-4d79-a288-e20b8c1f55d1
html"""<img src="https://github.com/GiggleLiu/NiLang.jl/raw/master/docs/src/asset/logo3.png"/>"""

# ╔═╡ 5d51231a-8bf0-4414-9a39-cea264df84f2
md"Initially written by Jinguo Liu and Taine Zhao (The author of MLStyle)"

# ╔═╡ e10e0be8-b26e-4719-92dc-8ca46af0b4b5
md"## Feature 1. one function for two"

# ╔═╡ b1a9946b-82b4-4954-8bb9-5df035eaefe4
md"Example: an identity mapping ``(x, y) \mapsto (x,y)`` "

# ╔═╡ 00342a51-36d8-4fdd-aab7-ee02e2122c49
@i function f1(x1, x2)
	# will return inputs automatically for you
end

# ╔═╡ 40c1c48d-0e5a-4a47-b7e4-8f7666281249
f1(2, 3)

# ╔═╡ 59df1f80-9be1-4b26-b263-ca0c7a0b9ab7
(~f1)(2, 3)

# ╔═╡ 8320d326-c1ab-4807-befb-13dda3480bf5
md"## Feature 2. every instruction is reversible, every object is ''mutable''"

# ╔═╡ 93238608-3b86-49f1-ad60-9360e12cff1c
md"General design patterns
* `y += f(x)`
* `y -= f(x)`
* `y ⊻= f(x)`

There are also instructions like `SWAP`, `ROT`."

# ╔═╡ f657c3fb-e140-4c76-8065-54f1cb6d05eb
md"Example: mutating fields of complex numbers"

# ╔═╡ 629a2549-745c-48a2-9bbc-a8f5fb046d11
@i function f2(x1::Complex, x2::Complex)
	x2 += exp(x1)       # accumulative form
	SWAP(x1.im, x2.im)  # other primitive functions
	f1(x1, x2)			# other reversible functions
end

# ╔═╡ 640e0029-7931-4afd-bdf9-fed317efbd8e
md" $(@bind expand_f2 CheckBox()) macroexpand"

# ╔═╡ 6930345b-6e93-4b35-8d4f-91ad49141fa1
if expand_f2
	macroexpand(NiLang, :(@i function f2(x1::Complex, x2::Complex)
		x2 += exp(x1)
		SWAP(x1.im, x2.im)
	end)) |> NiLangCore.rmlines
end

# ╔═╡ 090522bf-0ff2-4022-8460-aec6e37e936a
f2(1.0+2im, 2.0+4.9im)

# ╔═╡ 88c30609-2f42-405e-a14c-dfab44aef23b
(~f2)(f2(1.0+2im, 2.0+4.9im)...)

# ╔═╡ 2dc665a9-b131-4fef-acde-db346eb0f48b
md"## Feature 3. One can reverse the control flows too"

# ╔═╡ 4e479f48-42cd-476d-8604-08ecbb503a90
md"""
#### Reversible `if` statement
"""

# ╔═╡ dc41e99a-f598-4bf6-9f76-ecdb04f5f40c
leftright(md"
```julia
if (precondition[, postcondition])
	...
end
```
", md"
```julia
if (postcondition[, precondition])
	~(...)
end
```
")

# ╔═╡ 97e0bae1-69ac-4cbf-b9d9-6b38180edd78
TikzPicture(L"""
\node [test] (pre) {precondition};
\node [proc, it] (st1) [right=of pre] {statements 1};
\node [proc, it] (st2) {statements 2};	
\node [test] (post1) [right=of st1] {postcondition};
\node [test] (post2) [right=of st2] {postcondition};
\node [proc,red] (err1) [above=of post1] {invertibility error};
\node [proc,red] (err2) [below=of post2] {invertibility error};
\draw [->,black] (pre.east) -- (st1) node[midway,above] {T};
\draw [->,black] (pre.south) |- (st2) node[midway,below] {F};
\draw [->,black] (-2.5, 0.0) -- (pre.west);
\draw [->,black] (st1) -- (post1);
\draw [->,black] (st2) -- (post2);
\draw [->,red] (post1) -- (err1) node[midway,right] {F};
\draw [->,red] (post2) -- (err2) node[midway,right] {T};
\draw [->,black] (post1.east) -- (12, 0) node[midway,above] {T};
\draw [black] (post2.east) -| (11, 0) node[midway,right] {F};
""", options=raw"    font=\sffamily\small,
    >={Triangle[]},
    */.tip={Circle[]},
    start chain=going below,
    node distance=18mm and 40mm,
    every join/.style={norm},
    base/.style={draw, on chain, on grid, align=center, minimum height=4ex, inner color=black!50!gray!10, outer color=black!50!gray!15},
    proc/.style={base, rectangle, text width=8em},
    test/.style={base, diamond, text centered, aspect=2.6,inner sep=-0ex},
    norm/.style={->, draw, black},
    it/.style={font={\sffamily\small\itshape}}", preamble=raw"\usetikzlibrary{shapes.geometric,arrows.meta,chains,positioning,quotes}")

# ╔═╡ 355ba831-6be0-456a-8f94-36acd2365f17
md"Example: obtaining the absolute value ``x \mapsto |x|``"

# ╔═╡ 003c3e68-600e-4688-832b-5e061572b128
@i function abs_incorrect(x)
	if x < 0
		NEG(x)
	end
end

# ╔═╡ 1fb196f9-0f0a-42dc-b094-077cdf18d13d
abs_incorrect(-3)

# ╔═╡ aa9a679e-63bc-4951-b6f4-65316e212bc8
@i function abs_correct(x, sgn)
	if (x < 0, sgn)
		NEG(x)
		sgn ⊻= true
	end
end

# ╔═╡ e6fd3c2d-cadd-40d7-a575-b1e68c45ee13
abs_correct(-3, false)

# ╔═╡ 02b7e1b4-4622-4e68-966e-ff79817557d1
md"#### Reversible `while` statement"

# ╔═╡ 364fd613-0ebd-4b45-a3fd-f9baa8c487e3
leftright(md"
```julia
@from condition1 while condition2
	...
end
```
", md"
```julia
@from !(condition2) while !(condition1)
	~(...)
end
```
")

# ╔═╡ 75d8283a-b331-4648-84a8-489e168e33f9
TikzPicture(L"""
\node [test] (c1) {condition 1};
\node [test] (c2) [right=of c1]  {condition 2};
\node [test] (c3) [right=of c2]  {condition 1};
\node [proc, it] (st1) [above=of c2] {statements};
\node [proc,red] (err1) [below=of c1] {invertibility error};
\node [proc,red] (err2) [right=of c3] {invertibility error};
\draw [->,black] (c2) -- (st1) node[midway,right] {T};
\draw [->,black] (st1) -| (c3);
\draw [->,black] (-2.5, 0.0) -- (c1.west);
\draw [->,black] (c1) -- (c2) node[midway,above] {T};
\draw [->,black] (c3) -- (c2) node[midway,above] {F};
\draw [->,red] (c1) -- (err1) node[midway,right] {F};
\draw [->,red] (c3) -- (err2) node[midway,above] {T};
\draw [->,black] (c2.south) |- (11, -2) node[midway,below] {F};
""", options=raw"    font=\sffamily\small,
    >={Triangle[]},
    */.tip={Circle[]},
    start chain=going below,
    node distance=18mm and 40mm,
    every join/.style={norm},
    base/.style={draw, on chain, on grid, align=center, minimum height=4ex, inner color=black!50!gray!10, outer color=black!50!gray!15},
    proc/.style={base, rectangle, text width=8em},
    test/.style={base, diamond, text centered, aspect=2.6,inner sep=-0ex},
    norm/.style={->, draw, black},
    it/.style={font={\sffamily\small\itshape}}", preamble=raw"\usetikzlibrary{shapes.geometric,arrows.meta,chains,positioning,quotes}")

# ╔═╡ 2207c2fb-4a52-4766-8dd3-03872744aa74
md"example: computing Fibonacci numbers"

# ╔═╡ 288331c3-2dfb-4941-985f-554be409c0ab
@i function fib(y, n)
    @invcheckoff if (n >= 1, ~)
        counter ← 0
        counter += n
        @from counter==n while counter > 1
			counter -= 1
            fib(y, counter)
            counter -= 1
        end
        counter -= n % 2
        counter → 0
    end
    y += 1
end

# ╔═╡ 12a30359-d6ff-4113-bab5-b198e908cf1a
fib(0, 10)

# ╔═╡ 23ea88b4-6b89-462a-92da-0e8bdf5c73b5
(~fib)(89, 10)

# ╔═╡ 4bfdabd6-b7ff-40fb-b567-52910acb5a07
md"""
#### Reversible `for` statement

$(
leftright(md"
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

# ╔═╡ 2603147b-e7a2-4cae-b88f-2cfebe16bacb
md"## Feature 4. storage access should also be reversible"

# ╔═╡ 12d49e2e-cc6e-48d6-b11a-e7c311453bfc
md"""
$(
leftright(updown(md"
```julia
var ← zero(T)
```", md"borrow some memory from system and allocate it to variable var of type T."), updown(md"
```julia
var → zero(T)
```
", md"return the zero cleared variable to system."))
)
"""

# ╔═╡ 10312dc7-e861-4c89-b2fd-672cfe8850bf
md"""
$(
leftright(updown(md"
```julia
dict[key] ← variable
```", md"create a new entry, asserting `key` does not exist"), updown(md"
```julia
dict[key] → variable
```
", md"asserting the value of an existing key, and delete it."))
)
"""

# ╔═╡ 03f58f2a-24b6-4235-9102-71a19b9679ac
md"Example: implementing `y += log(x)` for complex number.

```math
\log(z) = \log(|z|) + i {\rm Arg}(z)
```"

# ╔═╡ 22a5853e-4f9a-4da0-bc03-84a6b0061cfe
@i function clog_v1(y::Complex{T}, squaren::T, n::T, x::Complex{T}) where T
	squaren += x.re^2
	squaren += x.im^2
	n += sqrt(squaren)
    y.re += log(n)
	y.im += atan(x.im, x.re)
end

# ╔═╡ 1ecdc4d2-b3ca-4f5c-a454-0f0bc51b6ec2
@test clog_v1(0.0im, 0.0, 0.0, 3.0im)[1] ≈ log(3.0im)

# ╔═╡ 927ea209-2ccd-48d5-b69e-0a3c735bb496
md"""Bennett, Charles H. "Logical reversibility of computation." (1973)."""

# ╔═╡ 962b204c-8195-4938-944c-b7c4a52e70bd
TikzPicture(L"""
\def\r{0.15};
\foreach \x in {1,...,5}{
	\fill[fill=black] (\x, 0) circle [radius=\r];
	\node[white] at (\x, 0) {$s_{\x}$};
}
\fill[fill=white] (5.5, 0) circle [radius=\r];
\foreach \x in {1,...,4}{
	\draw [black, thick, ->] (\x+\r, \r) .. controls (\x+0.5, 0.3) .. (\x+1-\r, \r);
	\node at (\x+0.5, 0.4) {\x};
	}
\foreach[evaluate={\y=int(8-\x)}] \x in {1,...,3}{
	\draw [red, thick, <-] (\x+\r, -\r) .. controls (\x+0.5, -0.3) .. (\x+1-\r, -\r);
	\node at (\x+0.5, -0.4) {\y};
	}
"""
, options="scale=2.0", preamble="")

# ╔═╡ af58a0f8-e3fd-465f-b1ae-6fbd94123c91
@i function clog_v2(y::Complex{T}, x::Complex{T}) where T
	######### compute ########
	n ← zero(T)
	squaren ← zero(T)
	squaren += x.re^2
	squaren += x.im^2
	n += sqrt(squaren)

	########## copy ##########
	
    y.re += log(n)
	y.im += atan(x.im, x.re)
	
	####### uncompute ########
	n -= sqrt(squaren)
	squaren -= x.im^2
	squaren -= x.re^2
	n → zero(T)
	squaren → zero(T)
end

# ╔═╡ 4f993061-c6c3-4acb-aef3-8453e7b83997
@test clog_v2(0.0im, 3.0im)[1] ≈ log(3.0im)

# ╔═╡ 6e5cf9bb-7cab-4da1-8831-541e0ee3bde8
@i @inline function clog_v3(y::Complex{T}, x::Complex{T}) where T
	# @invcheckoff turns of reversibility check and accelerate code
    @routine @invcheckoff begin
        @zeros T squaren n
		squaren += x.re^2
		squaren += x.im^2
		n += sqrt(squaren)
    end
    y.re += log(n)
    y.im += atan(x.im, x.re)
    ~@routine
end

# ╔═╡ 320e4114-f0da-4106-90ab-a9f7b0ef0099
@test clog_v3(0.0im, 3.0im)[1] ≈ log(3.0im)

# ╔═╡ 2d944e8d-e19d-48be-ab7f-c3e54e9f43ef
md"# III. Automatic differentiation in NiLang"

# ╔═╡ 4f53fa8e-ea9b-461f-8199-7bbe2a3ef544
md"## Scalar or tensor"

# ╔═╡ 56ea4f5f-ea88-46d6-beed-b9a7afed315d
md"Differentiating matrix vector multiplication"

# ╔═╡ 124a9ecd-0bda-4823-9507-92efcf449d9c
md"""
```math
y = A x
```
"""

# ╔═╡ 6819498c-7a46-48fa-9eec-38341dca72f9
let
	tl = md"tensor level view"
	tl2 = md"
```julia
y = A * x
```
"
	sl = md"scalar level view"
	sl2 = md"
```julia
for j=1:n
	for i=1:m
		y[i] += A[i,j] * x[j]
	end
end
```"
	leftright(updown(tl, tl2, width=300), updown(sl, sl2, width=300))
end

# ╔═╡ 400f79cf-9260-4195-9582-0e8c486ddb5a
html"""implementing AD on scalars
<ul>
<li style="color:green">limited primitive function</li>
<li style="color:red">hard to utilize BLAS</li>
<li style="color:red">harder to manage memory caching</li>
</ul>
"""

# ╔═╡ 2dad3acd-332c-46b9-8f84-21c076bdef41
md"## Forward mode autodiff and reverse mode autodiff"

# ╔═╡ a674bde5-70e1-4d21-aedf-8977f8039c36
md"
A program: ``\vec p \mapsto \vec q``, containing the following forward/backward instruction.

```math
\begin{cases}
\vec y = f(\vec x) \\
\vec x = f^{-1}(\vec y)
\end{cases}
```
"

# ╔═╡ bf696784-23d4-42b2-8ee1-bfec11ff8d78
let
	fd = md"""
```math
\begin{align}
    \frac{d \vec x}{d p_i} = \underbrace{\frac{d \vec y}{d \vec x}}_{\text{local jacobian}}\frac{d \vec x}{d p_i}
\end{align}
```


ForwardDiff: ``(\vec x, \frac{d\vec x}{dp_i}) \mapsto (y, \frac{d\vec y}{dp_i})``
"""

	bd = md"""
```math
\begin{align*}
    \frac{d q_j}{d \vec x} &\mathrel{+}= \frac{\partial q_j}{\partial \vec y}\underbrace{\frac{d \vec y}{d \vec x}}_{\text{local jacobian}}
\end{align*}
```
NiLang: ``(\vec y, \frac{d\mathcal{L}}{d\vec y}) \mapsto (\vec x, \frac{d\mathcal{L}}{d\vec x})``
"""
	leftright(fd, bd)
end

# ╔═╡ 9193fcbb-ec4f-41b4-8fca-be8e183dea31
g_forwarddiff = sin(Dual(π/3, 1.0))

# ╔═╡ 3fadf1d4-8fa2-4c02-aa21-b7969b465536
# note: y += sin(x) is translate to `PlusEq(sin)(y, x)` in NiLang.
g_nilang = MinusEq(sin)(GVar(sin(π/3), 1.0), GVar(π/3, 0.0))

# ╔═╡ 12e933ca-13f3-414c-926a-f1bb1bbe66cf
@test g_forwarddiff.partials[1] ≈ g_nilang[2].g

# ╔═╡ 4403c183-5eeb-4fd0-87a4-4ad29e1f4dc2
md"## Differentiating complex valued log"

# ╔═╡ 4c3f9b91-4f27-4fb8-a9db-1ddbfd62dbdd
@i function real_of_clog(loss::Real, y::Complex, x::Complex)
	clog_v3(y, x)
	loss += y.re
end

# ╔═╡ af71e2d7-a600-46fd-9a46-1b9d4607f06d
@test let
	# forward pass to compute results
	loss_out, y_out, x_out = real_of_clog(0.0, 0.0im, 2+3.0im)

	# backward pass to compute inputs from results, using element type `GVar`
	gloss_out = GVar(loss_out, 1.0)
	gy_out = GVar(y_out)
	gx_out = GVar(x_out)
	gloss_out, gy_out, gx_out = (~real_of_clog)(gloss_out, gy_out, gx_out)

	# forward diff
	dloss_out = Dual(loss_out, 0.0)
	dy_out = Complex(Dual(0.0, 0.0), Dual(0.0, 0.0))
	dx_out = Complex(Dual(2.0, 1.0), Dual(3.0, 0.0))
	dloss_out, dy_out, dx_out = real_of_clog(dloss_out, dy_out, dx_out)
	
	gx_out.re.g ≈ dloss_out.partials[1]
end

# ╔═╡ 1d3c7324-6828-47aa-b30e-bcac0e052213
md"A shortcut"

# ╔═╡ 2993fe1f-042b-4c85-85b8-bcc7ed449a54
NiLang.AD.gradient(real_of_clog, (0.0, 0.0im, 2+3.0im); iloss=1)

# ╔═╡ 048d482e-3e5b-496f-95d7-17589a5f6f11
md"# Overheads matters!"

# ╔═╡ e22bbe66-56f5-40be-b464-1f8651e6a6ac
md"""
* Case 1: Intrinsically irreversible linear program,
* Case 2: A linear algebra function: QR decomposition
"""

# ╔═╡ 28767d73-47a8-4f4b-b3dd-146c4ae3e038
md"## Case 1: differentiating a linear program"

# ╔═╡ ea907d51-d4a9-48ca-90a5-bd91309ccfad
md"Imagine we have a very long linear program that intrinsically irreversible"

# ╔═╡ 3aa99be5-6747-4163-9bb6-ed8cd5ce19f6
TikzPicture(L"""
\def\r{0.15};
\def\n{10};
\foreach \x in {\n}{
       \fill[fill=black] (\x, 0) circle [radius=\r];
       \node[white] at (\x, 0) {$s_{\x}$};
}
\foreach \x in {1,...,9}{
       \draw (\x, 0) circle [radius=\r];
       \node[black] at (\x, 0) {$s_{\x}$};
}
\fill[fill=white] (\n+0.5, 0) circle [radius=\r];
\foreach \x/\t in {1/1,2/2,3/3,4/4,5/5,6/6,7/7,8/8,9/9}{
       \draw [black, thick, ->] (\x+\r, \r) .. controls (\x+0.5, 0.3) .. (\x+1-\r, \r);
       \node[black] at (\x+0.5, 0.4) {\t};
       }
"""
, options="scale=2.0", preamble="")


# ╔═╡ 2f0263f5-2ace-4d4b-8d71-cee26c03122e
md"The accumulative version"

# ╔═╡ 03a21468-c2fe-4df7-ae8a-38b28a0efe2f
TikzPicture(L"""
\def\r{0.15};
\def\n{10};
\foreach \x in {1,...,\n}{
       \fill[fill=black] (\x, 0) circle [radius=\r];
       \node[white] at (\x, 0) {$s_{\x}$};
}
\fill[fill=white] (\n+0.5, 0) circle [radius=\r];
\foreach \x/\t in {1/1,2/2,3/3,4/4,5/5,6/6,7/7,8/8,9/9}{
       \draw [black, thick, ->] (\x+\r, \r) .. controls (\x+0.5, 0.3) .. (\x+1-\r, \r);
       \node[black] at (\x+0.5, 0.4) {\t};
       }
"""
, options="scale=2.0", preamble="")


# ╔═╡ f326d8e5-7117-4eed-b30d-0f64e5974426
md"With uncomputing"

# ╔═╡ b9af3db6-5725-4190-b96c-f3fa41f07c93
TikzPicture(L"""
\def\r{0.15};
\def\n{10};
\foreach \x in {1,4,7,10}{
       \fill[fill=black] (\x, 0) circle [radius=\r];
       \node[white] at (\x, 0) {$s_{\x}$};
}
\foreach \x in {2,3,5,6,8,9}{
       \draw (\x, 0) circle [radius=\r];
       \node[black] at (\x, 0) {$s_{\x}$};
}
\fill[fill=white] (\n+0.5, 0) circle [radius=\r];
\foreach \x/\t in {1/1,2/2,3/3,4/6,5/7,6/8,7/11,8/12,9/13}{
       \draw [black, thick, ->] (\x+\r, \r) .. controls (\x+0.5, 0.3) .. (\x+1-\r, \r);
       \node[black] at (\x+0.5, 0.4) {\t};
       }
\foreach \x/\t in {1/5,2/4,4/10,5/9,7/15,8/14}{
       \draw [black, thick, <-] (\x+\r, -\r) .. controls (\x+0.5, -0.3) .. (\x+1-\r, -\r);
       \node[black] at (\x+0.5, -0.4) {\t};
}
"""
, options="scale=2.0", preamble="")

# ╔═╡ 47cf7e85-c49f-4618-9689-e0de789625f6
md"With uncomputing: the coarser grain"

# ╔═╡ 2fcf48fd-bedc-41d9-a433-68efd5dd0d20
TikzPicture(L"""
\def\r{0.15};
\def\n{10};
\foreach \x in {1,4,7,10}{
       \fill[fill=black] (\x, 0) circle [radius=\r];
       \node[white] at (\x, 0) {$s_{\x}$};
}

\fill[fill=white] (\n+0.5, 0) circle [radius=\r];
\foreach \x in {1,4,7}{
       \draw [black, thick, ->] (\x+\r, \r) .. controls (\x+1.5, 0.6) .. (\x+3-\r, \r);
       }
\foreach \x in {1,4}{
       \draw [black, thick, <-] (\x+\r, -\r) .. controls (\x+1.5, -0.6) .. (\x+3-\r, -\r);
}
"""
, options="scale=2.0", preamble="")

# ╔═╡ 5060f5bb-8430-42aa-b61d-c88249edb323
md"## Pebble game"

# ╔═╡ 55dd53ed-6420-4426-95ae-feb47bf50f22
md"The optimal time-space tradeoff corresponds to the optimal solution to the pebble game."

# ╔═╡ c62a9f94-457f-496b-bee0-bb0db02aca5d
TikzPicture(L"""
\def\y{0}
\node at (4, \y-1) {initial configuration};
\foreach \x in {0,...,16}{
	\draw (0.5*\x-0.25, 0.5*\y-0.25) rectangle (0.5*\x+0.25, 0.5*\y+0.25);
	\ifnum \x > 0
		\node at (0.5*\x, 0.5*\y) {\x};
	\fi
}
\fill (0, 0)  ellipse (0.2 and 0.15);
\def\dx{11}
\foreach \a/\b in {-0.1/0.3, 0.2/0.5, -0.5/0.4, 0.1/0.24, 0.6/0.1, -0.3/-0.3}
	\fill (\dx+\a, \y+\b)  ellipse (0.2 and 0.15);
\node at (11, -1) {free pool of pebbles};
\node at (13, -1) {};
\node (goal) at (9, \y+1) {goal};
\draw[<-,thick] (8, \y+0.3) .. controls (8.2, \y+0.7) .. (goal);
""", options="scale=1.0")

# ╔═╡ bccadcb5-6d9f-4a70-b7c4-74e0e3d5f8c8
TikzPicture(L"""
\def\y{0}
\node at (4, \y-1) {put rule (if and only if the previous grid is occupied)};
\foreach \x in {0,...,16}
	\draw (0.5*\x-0.25, 0.5*\y-0.25) rectangle (0.5*\x+0.25, 0.5*\y+0.25);
\fill (0, 0)  ellipse (0.2 and 0.15);
\fill (2, 0)  ellipse (0.2 and 0.15);
\draw[dashed] (2.5, 0)  ellipse (0.2 and 0.15);
\def\dx{11}
\foreach \a/\b in {-0.1/0.3, 0.2/0.5, -0.5/0.4, 0.1/0.24, 0.6/0.1, -0.3/-0.3}
	\fill (\dx+\a, \y+\b)  ellipse (0.2 and 0.15);
\node at (13, -1) {};
\draw[<-,thick] (2.5, \y+0.3) .. controls (2.8, \y+1) and (8.0, \y+1) .. (10, 0.5);
""", options="scale=1.0")

# ╔═╡ b308ecb0-070e-4ad8-8009-dc60e75bbe01
TikzPicture(L"""
\def\y{0}
\node at (4, \y-1) {remove rule (if and only if the previous grid is occupied)};
\foreach \x in {0,...,16}
	\draw (0.5*\x-0.25, 0.5*\y-0.25) rectangle (0.5*\x+0.25, 0.5*\y+0.25);
\fill (0, 0)  ellipse (0.2 and 0.15);
\fill (2, 0)  ellipse (0.2 and 0.15);
\fill (2.5, 0)  ellipse (0.2 and 0.15);
\def\dx{11}
\foreach \a/\b in {-0.1/0.3, 0.2/0.5, -0.5/0.4, 0.1/0.24, 0.6/0.1, -0.3/-0.3}
	\fill (\dx+\a, \y+\b)  ellipse (0.2 and 0.15);
\node at (13, -1) {};
\draw[->,thick] (2.5, \y+0.3) .. controls (2.8, \y+1) and (8.0, \y+1) .. (10, 0.5);
""", options="scale=1.0")

# ╔═╡ 9123e669-19c7-47c1-a924-0c618b4a9c1f
md"
Space complexity: ``O(\log(T)S)``

Time complexity: ``O(T^{1+\epsilon})``
"

# ╔═╡ 83d0c5fc-9cb7-4e37-bfdb-ed630b94d9b4
md"
The recursive Bennett's time-space tradeoff scheme is probably optimal for a reversible program. (Li 1997)
"

# ╔═╡ 3d2feca5-43d3-4a46-ba1c-849c5ceeb676
md"nstep = $(@bind nstep Slider(2:20000; show_value=true, default=10000))"

# ╔═╡ 7ca616d1-caed-40cf-b768-b07af81a654d
md"k = $(@bind bennett_k Slider(2:100; show_value=true, default=2))"

# ╔═╡ a54d5fc4-643c-47d2-be97-4626a060c9b4
md"Optimal checkpointing is recursive, Griewank (1992). 
Julia Implementation: [https://github.com/GiggleLiu/TreeverseAlgorithm.jl](https://github.com/GiggleLiu/TreeverseAlgorithm.jl)"

# ╔═╡ 8b556fbe-275f-4e7b-94a0-7434ce81ad8b
md"Time complexity is ``O(T\log(T))``"

# ╔═╡ 8e9c55f6-8175-4cb4-8798-91107c4d16ee
md"Space complexity is ``O(S\log(T))``"

# ╔═╡ 972d889c-c48c-470a-b710-aba9ecaacdaa
md"nstep = $(@bind treeverse_nstep Slider(2:20000; show_value=true, default=10000))"

# ╔═╡ 54b3a283-7d42-431f-9ecb-48f37198409a
md"number of checkpoints = $(@bind treeverse_δ Slider(2:100; show_value=true, default=2))"

# ╔═╡ 590f56f7-5654-4493-9103-02a0fce6e945
let
	logger = TreeverseLog()
	treeverse(identity, (x,gy)->0, 0; δ=treeverse_δ, N=treeverse_nstep, logger=logger)
	logger
end

# ╔═╡ 11964529-8e88-4743-a725-57fc5c525649
html"""
<img src="https://user-images.githubusercontent.com/6257240/122655250-07346e00-d11f-11eb-9620-983ef16019a3.png" width=600/>
"""

# ╔═╡ 45448141-214d-4e08-978e-5d1d25f763cd
md"## Case 2: differentiating linear algebra functions"

# ╔═╡ dfe3cc09-6e27-4166-9a4e-2dd22f1e08a2
md"The definition of QR factorization

```math
A = QR
```
"

# ╔═╡ 3d07e6d1-2964-4718-b7bc-114d82389aa4
md"We implement Householder QR"

# ╔═╡ c00cd648-d685-4a17-9ddf-39c06dd5f066
md"""
$Q = H_1H_2 \ldots H_n$
"""

# ╔═╡ 5390c3ba-1507-4db9-9972-8295cbe493bc
md"""
```math
\begin{align}
&H = 1-\beta vv^T
\end{align}
```
"""

# ╔═╡ 33a0bda1-c943-4792-bc3d-fbf1adf16d0a
let
	img = TikzPicture(L"""
\draw[->,thick] (0, 0) -- (1, 1);
\draw[->,thick] (0, 0) -- ({sqrt(2)}, 0);
\draw[thick,dashed] (0, 0) -- (1.5, {1.5*tan(22.5)});
\node at (1.5, 0) {$e$};
\node at (1.1, 1.1) {$x$};
\node at (1.4, {1.6*tan(22.5)}) {$v$};
\node at (2.6, 0.5) {$v = x-\|x\|_2 e$};
""", options="scale=2.0", preamble="")
	HTML("""<div align=center>$(html(img))</div>""")
end

# ╔═╡ a2e81e79-3f85-4eef-8639-f455f9165a25
md"Apply reflector, step $(@bind house_step NumberField(0:4))"

# ╔═╡ 0a04c470-8f5a-4b56-b2e2-5b32e3b944f3
let
	num(i, j) = let
		sym = j>=i || house_step < j ? raw"\times" : raw"0"
		if i>=house_step && j>=house_step
			sym = "\\color{red}{$sym}"
		end
		sym
	end
	elements = join([join([num(i,j) for j=1:5], " & ") for i=1:5], raw"\\\\")
	diag(i) = if i == house_step
		raw"\boldsymbol{\times}"
	else
		raw"\times"
	end
	Markdown.parse("""
```math
\\begin{align}
$(join(["H_$i" for i in house_step:-1:1], "")) A = \\left(\\begin{matrix}
$elements\\end{matrix}\\right)
\\end{align}
```
""")
end

# ╔═╡ dced36eb-3b84-4d08-8268-0ffb831e39b5
md"the following code is adapted from the Julia standard library"

# ╔═╡ 0dc268fa-2b71-4915-b73b-3f1de9fbc157
struct Reflector{T,RT,VT<:AbstractVector{T}}
    ξ::T
    normu::RT
    sqnormu::RT
    r::T
    y::VT   # reflector vector
end

# ╔═╡ 871f62a3-2c9c-445e-b1dc-4cba363fa604
# compute "Householder" reflector
@i function reflector!(R::Reflector{T,RT}, x::AbstractVector{T}) where {T,RT}
    @inbounds @invcheckoff if length(x) != 0
        R.ξ += x[1]
        R.sqnormu += abs2(R.ξ)
        for i = 2:length(x)
            R.sqnormu += abs2(x[i])
        end
        if !iszero(R.sqnormu)
            R.normu += sqrt(R.sqnormu)
            if real(R.ξ) < 0
                NEG(R.normu)
            end
            R.ξ += R.normu
            R.y[1] -= R.normu
            for i = 2:length(x)
                R.y[i] += x[i] / R.ξ
            end
            R.r += R.ξ/R.normu
        end
    end
end

# ╔═╡ 6a50f7ac-b5a5-444d-9042-81b82cb66aec
# apply reflector from left
@i function reflectorApply!(vA::AbstractVector{T}, x::AbstractVector, τ::Number, A::StridedMatrix{T}) where T
    (m, n) ← size(A)
    @safe if length(x) != m || length(vA) != n
        throw(DimensionMismatch("reflector has length ($(length(x)), $(length(vA))), which must match the first dimension of matrix A, ($m, $n)"))
    end
    @inbounds @invcheckoff if m != 0
        for j = 1:n
            # dot
            @routine @zeros T vAj vAj_τ
            vAj += A[1, j]
            for i = 2:m
                vAj += x[i]'*A[i, j]
            end
            @routine vAj_τ += τ' * vAj  # `vAj_τ` can be uncomputed easily
            # ger
            A[1, j] -= vAj_τ
            for i = 2:m
                A[i, j] -= x[i]*vAj_τ
            end
            ~@routine
            SWAP(vA[j], vAj)
            ~@routine
        end
    end
    (m, n) → size(A)
end

# ╔═╡ bace7924-3aff-463f-9351-cc59191b469a
struct QRPivotedRes{T,RT,VT}
    factors::Matrix{T}                       # resulting matrix
    τ::Vector{T}
    jpvt::Vector{Int}                        # pivot vector
    reflectors::Vector{Reflector{T,RT,VT}}   # ~ half size of A (overhead)
    vAs::Vector{Vector{T}}         			 # ~ half size of A (overhead)
    jms::Vector{Int}
end

# ╔═╡ b12bcfbb-c7f2-4dc9-821a-e59365bf6fa4
begin
	_norm(v) = sqrt(sum(x->abs2(NiLang.value(x)), v))
	function indmaxcolumn(A::AbstractMatrix)
		mm = _norm(view(A, :, 1))
		ii = 1
		for i = 2:size(A, 2)
			mi = _norm(view(A, :, i))
			if abs(mi) > mm
				mm = mi
				ii = i
			end
		end
		return ii
	end
end;

# ╔═╡ 12d52dc8-da0b-46dc-9251-eb0cc9a39a7e
begin
	function alloc_qr(A::AbstractMatrix{T}) where T
		(m, n) = size(A)
		τ = zeros(T, min(m,n))
		jpvt = collect(1:n)
		reflectors = Reflector{T,real(T),Vector{T}}[]
		vAs = Vector{T}[]
		jms = Int[]
		QRPivotedRes(zero(A), τ, jpvt, reflectors, vAs, jms)
	end
	function alloc_reflector(x::AbstractVector{T}) where T
		RT = real(T)
		Reflector(zero(T), zero(RT), zero(RT), zero(T), zero(x))
	end
end

# ╔═╡ 923a5e2a-cc91-4404-913a-6e8012fa5834
@i function qr_pivoted!(res::QRPivotedRes, A::StridedMatrix{T}) where T
    m, n ← size(A)
    res.factors += A
    @inbounds @invcheckoff for j = 1:min(m,n)
        # Find column with maximum norm in trailing submatrix
        jm ← indmaxcolumn(view(res.factors, j:m, j:n)) + j - 1

		# pivot columns
        if jm != j
            # Flip elements in pivoting vector
            SWAP(res.jpvt[jm], res.jpvt[j])
            # Update matrix
			SWAP.(res.factors |> subarray(:, jm), res.factors |> subarray(:, j))
        end

        # Compute reflector of columns j
        R ← alloc_reflector(res.factors |> subarray(j:m, j))
        vA ← zeros(T, n-j)
        reflector!(R, res.factors |> subarray(j:m, j))
        # Update trailing submatrix with reflector
        reflectorApply!(vA, R.y, R.r, res.factors |> subarray(j:m, j+1:n))
        for i=1:length(R.y)
            SWAP(R.y[i], res.factors[j+i-1, j])
        end
		res.reflectors[end+1] ↔ R  # stack push
		res.vAs[end+1] ↔ vA
		res.jms[end+1] ↔ jm
    end
    @inbounds for i=1:length(res.reflectors)
        res.τ[i] += res.reflectors[i].r
    end
    m, n → size(A)
end

# ╔═╡ f8616bf4-02e6-4d66-a0f6-d35db701e82c
@testset "qr" begin
    for A in [randn(5, 5), randn(6, 4), randn(4, 6)]
        res = alloc_qr(A)
        res, = qr_pivoted!(res, copy(A))
        res2 = LinearAlgebra.qrfactPivotedUnblocked!(copy(A))
        @test res.factors ≈ res2.factors
        @test res.τ ≈ res2.τ
        @test res.jpvt ≈ res2.jpvt
    end
end

# ╔═╡ bee4ab06-4c60-4bec-89d0-b3c9a512f7a6
md"## Comparing with manual AD

"

# ╔═╡ 717f51fb-bd0a-4bb4-a5fc-1f1cf5d56ed8
html"""
<ul>
<li style="color:red">slower,</li>
<li style="color:red">an extra space overhead of the size of input matrix,</li>
<li style="color:green">stabler, e.g. can handle rank deficient matrices</li>
<li style="color:green">works consistently for complex numbers.</li>
</ul>
"""

# ╔═╡ db6b6481-7e67-4418-b907-13b38c77bac7
md"## Performance"

# ╔═╡ a0be4807-52ed-4626-8009-97a79e36e2f1
let  # Note: approximately 2x slower than the BLAS version
	Random.seed!(3)
	A = randn(ComplexF64, 200, 200)
	@benchmark LinearAlgebra.qrfactPivotedUnblocked!(copy($A))
end

# ╔═╡ 52dec342-d3de-4a88-8acb-0aa186bcc086
let
	Random.seed!(3)
	A = randn(ComplexF64, 200, 200)
	@benchmark qr_pivoted!(alloc_qr($A), copy($A))
end

# ╔═╡ 280c9363-9b49-44f1-a240-b7f205ffc56b
@benchmark let
	res = alloc_qr(A)
	res, A = qr_pivoted!(res, A)
	(~qr_pivoted!)(GVar(res), GVar(A))
end setup=(Random.seed!(3); A = randn(ComplexF64, 200, 200))

# ╔═╡ 0b3735c2-695c-4225-843e-16ca17aac0eb
md"""## Take home message

1. Comparing to irreversible computing, reversible computing is more **energy efficient**.
2. Reversible programming suffers from **polynomial time overhead and logarithmic space overhead** when differentiating a irreversible linear program. The overhead is much less when writting linear algebra functions.
3. Reversible embedded domain specific language NiLang.jl:
$(html"<div align=center><img  src='https://github.com/GiggleLiu/NiLang.jl/raw/master/docs/src/asset/logo3.png' width=300/></div>")
2. It is easy to balance time and space when differentiating a program in a reversible programming language.
4. It is not always more reliable to differentiate a program by deriving the backward rule manually.

5. TODOs
    - Is it possible to port `LoopVectorization` to NiLang to write blas level reversible program?
6. **How to find this notebook?** In NiLang's Github repo, file: `notebooks/reversibleprog.jl`
"""

# ╔═╡ d7942b37-f821-494a-8f18-5f267aa3457a
md"""
###  References
* Reeb, David, and Michael M. Wolf. "An improved Landauer principle with finite-size corrections." (2014).
* Athas, William C., and L. J. Svensson. "Reversible logic issues in adiabatic CMOS." Proceedings Workshop on Physics and Computation. (1994).
* Takeuchi, N., Y. Yamanashi, and N. Yoshikawa. "Reversible logic gate using adiabatic superconducting devices." (2014)
* Griewank, Andreas. "Achieving logarithmic growth of temporal and spatial complexity in reverse automatic differentiation." Optimization Methods and software 1.1 (1992): 35-54.
* Ming Li, John Tromp, Paul Vitanyi. "Reversible Simulation of Irreversible Computation by Pebble Games" (1997)
"""

# ╔═╡ 865f049f-54d0-4d21-860e-062262edcb58
md"# Removed"

# ╔═╡ c3b730a4-d5b4-471e-bd06-30ace6e8b8fe
let
	nodes_list = [[0], [0,1], [0,1,2], [0,2], [0,2,3], [0,2,3,4], [0,2,4], [0,1,2,4], [0,1,4], [0,4], [0,4,5], [0,4,5,6], [0,4,6], [0,4,6,7], [0,4,6,7,8], [0,4,6,8], [0,4,5,6,8], [0,4,5,8], [0,4,8],  [0,1,4,8], [0,1,2,4,8], [0,2,4,8], [0,2,3,4,8], [0,2,3,8], [0,2,8], [0,1,2,8], [0,1,8], [0,8],[0,8,9], [0,8,9,10], [0,8,10], [0,8,10,11], [0,8,10,11,12], [0,8,10,12], [0,8,9,10,12], [0,8,9,12], [0,8,12], [0,8,12,13], [0,8,12,13,14], [0,8,12,14], [0,8,12,14,15], [0,8,12,14,15,16]]
	s = join([raw"""
	\def\y{"""*string(j-1)*raw"""}
	\node at (-1, -0.7*\y) {step \y};
	\foreach \x in {0,...,16}{
		\draw (0.5*\x-0.25, -0.7*\y-0.25) rectangle (0.5*\x+0.25, -0.7*\y+0.25);
	}
	\foreach \x in {"""*join(nodes, ",")*raw"""}{
		\fill (0.5*\x, -0.7*\y)  ellipse (0.2 and 0.15);
	}
""" for (j, nodes) in enumerate(nodes_list)], "\n")
	TikzPicture(LaTeXString(s), options="scale=1.0")
end

# ╔═╡ 98f42f60-7870-4813-b0c1-728285c25f01
md"## Finding maximum, the reversible programming implementation"

# ╔═╡ c3c63865-f538-4d93-bef3-6b9c69cb177f
md"The naive implementation with linear space overhead"

# ╔═╡ 66225c05-165e-4051-bb5e-4cfba579fd5b
@i function i_find_maximum(m, y::AbstractVector, x::AbstractVector) where T
	@safe @assert !isempty(x) && length(y) == length(x)   # error handling
	y[1] += x[1]
	for i=2:length(x)
		if x[i] > y[i-1]
			y[i] += x[i]
		else
			y[i] += y[i-1]
		end
	end
	m += y[end]
end

# ╔═╡ 14291e8d-001f-4e26-b094-addb970cf530
x = randn(17)

# ╔═╡ bf670e51-06f8-4094-949c-ca2d02fd0d01
find_maximum(x)

# ╔═╡ 416e53e4-6123-430f-b433-4334b7e85298
i_find_maximum(0.0, zero(x), x)

# ╔═╡ edd5f8df-9abd-4254-abf6-33ae31d88a8d
struct FindMaxState{T}
	m::T
	step::Int
end

# ╔═╡ c0ff9103-2aa8-45fd-bea5-1903c53196de
let
	x, y = FindMaxState(5.0, 1), FindMaxState(2.0, 3)
	@instr x += y
	x, y
end

# ╔═╡ e17e70cb-2d82-4a08-9f6e-e50dcaa325e3
@i function i_find_maximum_step(t, s, x)
	t.step += s.step + 1
	if x[t.step] > s.m
		t.m += x[t.step]  # everything is mutable in NiLang
	else
		t.m += s.m
	end
end

# ╔═╡ ddc662d7-6901-4885-9d2c-1876a2c9d2ff
let
	Random.seed!(3)
	x = randn(nstep)
	logger = NiLang.BennettLog()
	output = NiLang.bennett(i_find_maximum_step, FindMaxState(0.0, 0), FindMaxState(x[1], 1), x; k=bennett_k, N=length(x)-1, logger=logger)[2]
	Text("output = $(output.m)\n\n$logger")
end

# ╔═╡ d2001eb2-45cb-4f07-aa99-dd84996359b7
md"## The connection to automatic differentiation"

# ╔═╡ fa3b6d6a-a55d-4097-8ad2-7dafb5f01d8c
md"""
* put rule: Only if there exists a pebble in grid $i$, you can move a pebble from your own pool to the grid $i+1$,
* take rule: you can take a pebble from the board any time,
* doodle rule: you can doodle grid $i$ only it when this grid has a pebble in it and grid $i+1$ is doodled,
* end rule: doodle all grids.
"""

# ╔═╡ dabb4656-4aed-4168-8659-c0472528c41d
md"""
## Optimal time
"""

# ╔═╡ 0367be06-4185-4add-a04b-f696c5a43638
TikzPicture(L"""
\foreach[evaluate={\j=int(16-\y)}] \y in {0,...,16}{
	\node at (-1, 0.7*\y) {step \j};
	\foreach \x in {0,...,16}{
		\draw (0.5*\x-0.25, 0.7*\y-0.25) rectangle (0.5*\x+0.25, 0.7*\y+0.25);
	}
	\foreach \x in {0,...,\j}{
		\fill (0.5*\x, 0.7*\y)  ellipse (0.2 and 0.15);
	}
}
""", options="scale=1.0")

# ╔═╡ c79e7651-975b-407c-8c8c-d0c5653ec570
md"
Space complexity: ``O(TS)``

Time complexity: ``O(T)``
"

# ╔═╡ 3fed55d7-dbed-4fc9-8410-2633d5200db6
md"""
## Limited space
"""

# ╔═╡ 663180d2-8fe1-4996-a194-d38120ae05fd
md"The recursive Bennett's time-space tradeoff"

# ╔═╡ 39884e8a-bc83-4ff4-85bc-cfaccb4674f2
html"""
<img src="https://user-images.githubusercontent.com/6257240/123340553-5cef8880-d51a-11eb-98f4-d402fa6b6532.png" width=500/>
"""

# ╔═╡ c24e7391-a187-4a7e-aab7-93027f7db965
md"The space optimal solution for 16 grids requires recursive Bennett's algorithm"

# ╔═╡ 12734842-d66f-4bfc-a9ad-01adeb2450e0
# stepfunc: step function
# state: state dictionary, initial value should contain entry `state[base]`
# k: compute `k` chunks forward and `k-1` chunks backward
# base: starting point
# len: number of steps to compute
@i function bennett_alg!(stepfunc, state::Dict{Int,T}, k::Int, base, len, args...; kwargs...) where T
    if len == 1  		# lowest level
        state[base+1] ← _zero(state[base])
        stepfunc(state[base+1], state[base], args...; kwargs...)
    else
		@safe @assert len % k == 0
        @routine begin  # compute block size
            chunksize ← 0
			start ← 0
			chunksize += len ÷ k
			start += base
			for j=1:k-1
				bennett_alg!(stepfunc, state, k, start, chunksize, args...; kwargs...)
				start += chunksize
			end
		end
		bennett_alg!(stepfunc, state, k, start, chunksize, args...; kwargs...)
        ~@routine
    end
end

# ╔═╡ 357fd442-6e6e-4b14-b2d0-efd3fa775d0b
bennett_alg!(i_find_maximum_step, Dict(0=>FindMaxState(x[1], 1)), 2, 0, length(x)-1, x)[2]

# ╔═╡ 1bca53ef-3438-4db5-9900-9fee71936a62
@i function loss(result, state, x)
	nstep ← length(x)-1
	bennett_alg!((@skip! i_find_maximum_step), state, 2, 0, nstep, x)
	result += state[nstep].m
	nstep → length(x)-1
end

# ╔═╡ 4e273d1f-a00e-49ca-9f8d-a0f6930550fb
let
	@testset "qr pivoted gradient" begin
		Random.seed!(3)
		A = randn(ComplexF64, 5, 5)
		res = alloc_qr(A)
		res, = qr_pivoted!(res, copy(A))
		res2 = LinearAlgebra.qrfactPivotedUnblocked!(copy(A))
		@test res.factors ≈ res2.factors
		@test res.τ ≈ res2.τ
		@test res.jpvt ≈ res2.jpvt

		# rank deficient initial matrix
		n = 50
		U = LinearAlgebra.qr(randn(n, n)).Q
		Σ = Diagonal((x=randn(n); x[n÷2+1:end] .= 0; x))
		A = U*Σ*U'
		res = alloc_qr(A)
		@test rank(A) == n ÷ 2
		qrres = qr_pivoted!(deepcopy(res), copy(A))[1]
		@test count(x->(x>1e-12), sum(abs2, QRPivoted(qrres.factors, qrres.τ, qrres.jpvt).R, dims=2)) == n ÷ 2

		#A = randn(ComplexF64, n, n)
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

# ╔═╡ 50f7070d-9f6f-4025-9be3-13812c3000eb
let
	x = [1.0, 3.0, 2.0, 1.3, -1.0]
	NiLang.gradient(loss, (0.0, Dict(0=>FindMaxState(x[1], 1)), x); iloss=1)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Compose = "a81c6b42-2e10-5240-aca2-a61377ecd94b"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
NiLang = "ab4ef3a6-0b42-11ea-31f6-e34652774712"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Revise = "295af30f-e4ad-537b-8983-00126c2a3abe"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
TikzPictures = "37f6aa50-8035-52d0-81c2-5a1d08754b2d"
TreeverseAlgorithm = "e1c63c57-2fea-45bf-a8bf-df3ea6afb545"
Viznet = "52a3aca4-6234-47fd-b74a-806bdf78ede9"

[compat]
BenchmarkTools = "~1.0.0"
Compose = "~0.9.2"
ForwardDiff = "~0.10.18"
NiLang = "~0.9.1"
PlutoUI = "~0.7.9"
Revise = "~3.1.17"
TikzPictures = "~3.3.3"
TreeverseAlgorithm = "~0.1.0"
Viznet = "~0.3.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Statistics", "UUIDs"]
git-tree-sha1 = "01ca3823217f474243cc2c8e6e1d1f45956fe872"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.0.0"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "be770c08881f7bb928dfd86d1ba83798f76cf62a"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "0.10.9"

[[CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "8ad457cfeb0bca98732c97958ef81000a543e73e"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.0.5"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dc7dedc2c2aa9faf59a55c622760a25cbefbe941"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.31.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Compose]]
deps = ["Base64", "Colors", "DataStructures", "Dates", "IterTools", "JSON", "LinearAlgebra", "Measures", "Printf", "Random", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "c6461fc7c35a4bb8d00905df7adafcff1fe3a6bc"
uuid = "a81c6b42-2e10-5240-aca2-a61377ecd94b"
version = "0.9.2"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4437b64df1e0adccc3e5d1adbc3ac741095e4677"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.9"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Dierckx]]
deps = ["Dierckx_jll"]
git-tree-sha1 = "5fefbe52e9a6e55b8f87cb89352d469bd3a3a090"
uuid = "39dd38d3-220a-591b-8e3c-4c3a8c710a94"
version = "0.5.1"

[[Dierckx_jll]]
deps = ["CompilerSupportLibraries_jll", "Libdl", "Pkg"]
git-tree-sha1 = "a580560f526f6fc6973e8bad2b036514a4e3b013"
uuid = "cd4c43a9-7502-52ba-aa6d-59fb2a88580b"
version = "0.0.1+0"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "214c3fcac57755cfda163d91c58893a8723f93e9"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.0.2"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "NaNMath", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "e2af66012e08966366a43251e1fd421522908be6"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.18"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "47ce50b742921377301e15005c96e979574e130b"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.1+0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "81690084b6198a2e1da36fcfda16eeca9f9f24e4"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.1"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "31c2eee64c1eee6e8e3f30d5a03d4b5b7086ab29"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.8.18"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LittleCMS_jll]]
deps = ["JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg"]
git-tree-sha1 = "e6ea89d915cdad8d264f7f9158c6664f879edcde"
uuid = "d3a379c0-f9a3-5b72-a4c0-6bf4d2e8af0f"
version = "2.9.0+0"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "LinearAlgebra"]
git-tree-sha1 = "1ba664552f1ef15325e68dc4c05c3ef8c2d5d885"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.2.4"

[[LogarithmicNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "d88b70111754e3660f80d3596a343ce42bf5ee84"
uuid = "aa2f6b4e-9042-5d33-9679-40d3a6b85899"
version = "0.4.2"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "4bfb8b57df913f3b28a6bd3bdbebe9a50538e689"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.1.0"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "6a8a2a625ab0dea913aba95c11370589e0239ff0"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.6"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MatchCore]]
git-tree-sha1 = "90af9fe333f8c9851f952dfa7f335185c94567c0"
uuid = "5dd3f0b1-72a9-48ad-ae6e-79f673da005f"
version = "0.1.1"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[NiLang]]
deps = ["FixedPointNumbers", "LinearAlgebra", "LogarithmicNumbers", "MatchCore", "NiLangCore", "Reexport", "SparseArrays", "TupleTools"]
git-tree-sha1 = "3fe439482d8c08a15f929ae7278a6c7f737672d5"
uuid = "ab4ef3a6-0b42-11ea-31f6-e34652774712"
version = "0.9.1"

[[NiLangCore]]
deps = ["MatchCore", "TupleTools"]
git-tree-sha1 = "239f97ea947531cfe7a596746e31c8429c7169b9"
uuid = "575d3204-02a4-11ea-3f62-238caa8bf11e"
version = "0.10.3"

[[OpenJpeg_jll]]
deps = ["Libdl", "Libtiff_jll", "LittleCMS_jll", "Pkg", "libpng_jll"]
git-tree-sha1 = "e330ffff1c6a593fa44cc40c29900bee82026406"
uuid = "643b3616-a352-519d-856d-80112ee9badc"
version = "2.3.1+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "c8abc88faa3f7a3950832ac5d6e690881590d6dc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.0"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "44e225d5837e2a2345e69a1d1e01ac2443ff9fcb"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.9"

[[Poppler_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "OpenJpeg_jll", "Pkg", "libpng_jll"]
git-tree-sha1 = "e11443687ac151ac6ef6699eb75f964bed8e1faa"
uuid = "9c32591e-4766-534b-9725-b71a8799265b"
version = "0.87.0+2"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "5f6c21241f0f655da3952fd60aa18477cf96c220"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.1.0"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "410bbe13d9a7816e862ed72ac119bda7fb988c08"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.1.17"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "a50550fa3164a8c46747e62063b4d774ac1bcf49"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.5.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "745914ebcd610da69f3cb6bf76cb7bb83dcb8c9a"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.4"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TikzPictures]]
deps = ["LaTeXStrings", "Poppler_jll", "Requires"]
git-tree-sha1 = "06b36e2baa9b97814ef1993207b71e2e23e9efb5"
uuid = "37f6aa50-8035-52d0-81c2-5a1d08754b2d"
version = "3.3.3"

[[TreeverseAlgorithm]]
deps = ["Requires"]
git-tree-sha1 = "4292bc608573c2047fd12b0a611787e77f5595ba"
uuid = "e1c63c57-2fea-45bf-a8bf-df3ea6afb545"
version = "0.1.0"

[[TupleTools]]
git-tree-sha1 = "62a7a6cd5a608ff71cecfdb612e67a0897836069"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.2.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Viznet]]
deps = ["Compose", "Dierckx"]
git-tree-sha1 = "7a022ae6ac8b153d47617ed8c196ce60645689f1"
uuid = "52a3aca4-6234-47fd-b74a-806bdf78ede9"
version = "0.3.3"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─e20e2d2e-4b28-4e32-8d80-ce029928a094
# ╟─f3e235e7-76b9-4c39-bc70-038539838ff4
# ╟─8308df59-3faa-4abf-8f05-119bbae48f64
# ╟─a3532a83-9fd3-4d24-b1bb-b52457317e51
# ╟─15657e4b-848e-43ad-a99f-37143d11705e
# ╟─34a6b7f4-7d72-485d-86dc-f4b1ba6174eb
# ╟─67d1b500-964e-4668-a7d0-ed93886446ca
# ╟─673992ed-6963-400a-a69b-d65d26c4f443
# ╟─6078758e-b392-4bdb-a1e7-44b135ce900e
# ╠═41e06e2a-b482-4e0f-8569-fee2ffd8aaaf
# ╟─dcf53d46-e259-4101-8530-9621094ee586
# ╟─6a88d26c-c895-4852-ab4f-37297b848731
# ╟─f9675365-36aa-430c-b747-3bc4f602e6fb
# ╟─46eb4ba9-dce6-4711-9c4d-3f16de6240de
# ╟─046f7559-4af9-4982-b5c3-335add0911d7
# ╟─0a039bfa-571e-4fad-b73c-1324d08777fc
# ╟─3f1e4d7a-32a7-4c7e-92dd-465bac925e63
# ╟─f68bcfb6-97ce-48d1-b0b8-e8466d4ac879
# ╟─3d4ba750-8d62-48ac-bf96-691397689ddc
# ╟─f4cb9212-181f-4338-b858-1d99c7f415e9
# ╟─31bde262-6352-4be0-b5cc-1781e3df2268
# ╟─83ff3fc3-bcd8-4235-a42f-1d75c7d6aa5b
# ╟─b308e270-6b40-4946-ac92-c705823f2c1e
# ╟─e483b3d4-d01c-4a98-8e68-e8120a7d95a7
# ╟─20c34526-c7c4-11eb-21fa-d706fd684a4c
# ╟─3f96abdf-fb5f-4d79-a288-e20b8c1f55d1
# ╟─5d51231a-8bf0-4414-9a39-cea264df84f2
# ╠═3b0fd2b5-5c6d-4d56-9e48-cda1493b4c72
# ╟─e10e0be8-b26e-4719-92dc-8ca46af0b4b5
# ╟─b1a9946b-82b4-4954-8bb9-5df035eaefe4
# ╠═00342a51-36d8-4fdd-aab7-ee02e2122c49
# ╠═40c1c48d-0e5a-4a47-b7e4-8f7666281249
# ╠═59df1f80-9be1-4b26-b263-ca0c7a0b9ab7
# ╟─8320d326-c1ab-4807-befb-13dda3480bf5
# ╟─93238608-3b86-49f1-ad60-9360e12cff1c
# ╟─f657c3fb-e140-4c76-8065-54f1cb6d05eb
# ╠═629a2549-745c-48a2-9bbc-a8f5fb046d11
# ╟─640e0029-7931-4afd-bdf9-fed317efbd8e
# ╟─6930345b-6e93-4b35-8d4f-91ad49141fa1
# ╠═090522bf-0ff2-4022-8460-aec6e37e936a
# ╠═88c30609-2f42-405e-a14c-dfab44aef23b
# ╟─2dc665a9-b131-4fef-acde-db346eb0f48b
# ╟─4e479f48-42cd-476d-8604-08ecbb503a90
# ╟─dc41e99a-f598-4bf6-9f76-ecdb04f5f40c
# ╟─97e0bae1-69ac-4cbf-b9d9-6b38180edd78
# ╟─355ba831-6be0-456a-8f94-36acd2365f17
# ╠═003c3e68-600e-4688-832b-5e061572b128
# ╠═1fb196f9-0f0a-42dc-b094-077cdf18d13d
# ╠═aa9a679e-63bc-4951-b6f4-65316e212bc8
# ╠═e6fd3c2d-cadd-40d7-a575-b1e68c45ee13
# ╟─02b7e1b4-4622-4e68-966e-ff79817557d1
# ╟─364fd613-0ebd-4b45-a3fd-f9baa8c487e3
# ╟─75d8283a-b331-4648-84a8-489e168e33f9
# ╟─2207c2fb-4a52-4766-8dd3-03872744aa74
# ╠═288331c3-2dfb-4941-985f-554be409c0ab
# ╠═12a30359-d6ff-4113-bab5-b198e908cf1a
# ╠═23ea88b4-6b89-462a-92da-0e8bdf5c73b5
# ╟─4bfdabd6-b7ff-40fb-b567-52910acb5a07
# ╟─2603147b-e7a2-4cae-b88f-2cfebe16bacb
# ╟─12d49e2e-cc6e-48d6-b11a-e7c311453bfc
# ╟─10312dc7-e861-4c89-b2fd-672cfe8850bf
# ╟─03f58f2a-24b6-4235-9102-71a19b9679ac
# ╠═22a5853e-4f9a-4da0-bc03-84a6b0061cfe
# ╠═1ecdc4d2-b3ca-4f5c-a454-0f0bc51b6ec2
# ╟─927ea209-2ccd-48d5-b69e-0a3c735bb496
# ╟─962b204c-8195-4938-944c-b7c4a52e70bd
# ╠═af58a0f8-e3fd-465f-b1ae-6fbd94123c91
# ╠═4f993061-c6c3-4acb-aef3-8453e7b83997
# ╠═6e5cf9bb-7cab-4da1-8831-541e0ee3bde8
# ╠═320e4114-f0da-4106-90ab-a9f7b0ef0099
# ╟─2d944e8d-e19d-48be-ab7f-c3e54e9f43ef
# ╟─4f53fa8e-ea9b-461f-8199-7bbe2a3ef544
# ╟─56ea4f5f-ea88-46d6-beed-b9a7afed315d
# ╟─124a9ecd-0bda-4823-9507-92efcf449d9c
# ╟─6819498c-7a46-48fa-9eec-38341dca72f9
# ╟─400f79cf-9260-4195-9582-0e8c486ddb5a
# ╟─2dad3acd-332c-46b9-8f84-21c076bdef41
# ╟─a674bde5-70e1-4d21-aedf-8977f8039c36
# ╟─bf696784-23d4-42b2-8ee1-bfec11ff8d78
# ╠═a7810352-7967-460d-abd7-361a324c20a9
# ╠═9193fcbb-ec4f-41b4-8fca-be8e183dea31
# ╠═a56445b5-e530-4035-9ac6-a2d196a6276a
# ╠═3fadf1d4-8fa2-4c02-aa21-b7969b465536
# ╠═12e933ca-13f3-414c-926a-f1bb1bbe66cf
# ╟─4403c183-5eeb-4fd0-87a4-4ad29e1f4dc2
# ╠═4c3f9b91-4f27-4fb8-a9db-1ddbfd62dbdd
# ╠═af71e2d7-a600-46fd-9a46-1b9d4607f06d
# ╟─1d3c7324-6828-47aa-b30e-bcac0e052213
# ╠═2993fe1f-042b-4c85-85b8-bcc7ed449a54
# ╟─048d482e-3e5b-496f-95d7-17589a5f6f11
# ╟─e22bbe66-56f5-40be-b464-1f8651e6a6ac
# ╟─28767d73-47a8-4f4b-b3dd-146c4ae3e038
# ╟─ea907d51-d4a9-48ca-90a5-bd91309ccfad
# ╟─3aa99be5-6747-4163-9bb6-ed8cd5ce19f6
# ╟─2f0263f5-2ace-4d4b-8d71-cee26c03122e
# ╟─03a21468-c2fe-4df7-ae8a-38b28a0efe2f
# ╟─f326d8e5-7117-4eed-b30d-0f64e5974426
# ╟─b9af3db6-5725-4190-b96c-f3fa41f07c93
# ╟─47cf7e85-c49f-4618-9689-e0de789625f6
# ╟─2fcf48fd-bedc-41d9-a433-68efd5dd0d20
# ╟─5060f5bb-8430-42aa-b61d-c88249edb323
# ╟─55dd53ed-6420-4426-95ae-feb47bf50f22
# ╟─c62a9f94-457f-496b-bee0-bb0db02aca5d
# ╟─bccadcb5-6d9f-4a70-b7c4-74e0e3d5f8c8
# ╟─b308ecb0-070e-4ad8-8009-dc60e75bbe01
# ╟─9123e669-19c7-47c1-a924-0c618b4a9c1f
# ╟─83d0c5fc-9cb7-4e37-bfdb-ed630b94d9b4
# ╠═c6bd40af-50ed-4cee-8043-60b2bac05058
# ╟─3d2feca5-43d3-4a46-ba1c-849c5ceeb676
# ╟─7ca616d1-caed-40cf-b768-b07af81a654d
# ╠═ddc662d7-6901-4885-9d2c-1876a2c9d2ff
# ╟─a54d5fc4-643c-47d2-be97-4626a060c9b4
# ╟─8b556fbe-275f-4e7b-94a0-7434ce81ad8b
# ╟─8e9c55f6-8175-4cb4-8798-91107c4d16ee
# ╠═873ef2c2-653e-425e-9732-b1ed19f7a0b7
# ╟─972d889c-c48c-470a-b710-aba9ecaacdaa
# ╟─54b3a283-7d42-431f-9ecb-48f37198409a
# ╠═590f56f7-5654-4493-9103-02a0fce6e945
# ╟─11964529-8e88-4743-a725-57fc5c525649
# ╟─45448141-214d-4e08-978e-5d1d25f763cd
# ╟─dfe3cc09-6e27-4166-9a4e-2dd22f1e08a2
# ╟─3d07e6d1-2964-4718-b7bc-114d82389aa4
# ╟─c00cd648-d685-4a17-9ddf-39c06dd5f066
# ╟─5390c3ba-1507-4db9-9972-8295cbe493bc
# ╟─33a0bda1-c943-4792-bc3d-fbf1adf16d0a
# ╟─a2e81e79-3f85-4eef-8639-f455f9165a25
# ╟─0a04c470-8f5a-4b56-b2e2-5b32e3b944f3
# ╟─dced36eb-3b84-4d08-8268-0ffb831e39b5
# ╠═141e21c0-1bdf-4e6b-b76d-129567a1180f
# ╠═0dc268fa-2b71-4915-b73b-3f1de9fbc157
# ╠═871f62a3-2c9c-445e-b1dc-4cba363fa604
# ╠═6a50f7ac-b5a5-444d-9042-81b82cb66aec
# ╠═bace7924-3aff-463f-9351-cc59191b469a
# ╠═923a5e2a-cc91-4404-913a-6e8012fa5834
# ╟─b12bcfbb-c7f2-4dc9-821a-e59365bf6fa4
# ╠═12d52dc8-da0b-46dc-9251-eb0cc9a39a7e
# ╠═ac35b26c-0585-4d2a-8fbf-bda9b141d6af
# ╠═f8616bf4-02e6-4d66-a0f6-d35db701e82c
# ╠═4e273d1f-a00e-49ca-9f8d-a0f6930550fb
# ╟─bee4ab06-4c60-4bec-89d0-b3c9a512f7a6
# ╟─717f51fb-bd0a-4bb4-a5fc-1f1cf5d56ed8
# ╟─db6b6481-7e67-4418-b907-13b38c77bac7
# ╠═d4726239-81af-4792-8472-c680508449c6
# ╠═a0be4807-52ed-4626-8009-97a79e36e2f1
# ╠═52dec342-d3de-4a88-8acb-0aa186bcc086
# ╠═280c9363-9b49-44f1-a240-b7f205ffc56b
# ╟─0b3735c2-695c-4225-843e-16ca17aac0eb
# ╟─d7942b37-f821-494a-8f18-5f267aa3457a
# ╟─865f049f-54d0-4d21-860e-062262edcb58
# ╟─c3b730a4-d5b4-471e-bd06-30ace6e8b8fe
# ╟─98f42f60-7870-4813-b0c1-728285c25f01
# ╟─c3c63865-f538-4d93-bef3-6b9c69cb177f
# ╠═66225c05-165e-4051-bb5e-4cfba579fd5b
# ╠═14291e8d-001f-4e26-b094-addb970cf530
# ╠═bf670e51-06f8-4094-949c-ca2d02fd0d01
# ╠═416e53e4-6123-430f-b433-4334b7e85298
# ╠═edd5f8df-9abd-4254-abf6-33ae31d88a8d
# ╠═c0ff9103-2aa8-45fd-bea5-1903c53196de
# ╠═e17e70cb-2d82-4a08-9f6e-e50dcaa325e3
# ╠═357fd442-6e6e-4b14-b2d0-efd3fa775d0b
# ╟─d2001eb2-45cb-4f07-aa99-dd84996359b7
# ╠═1bca53ef-3438-4db5-9900-9fee71936a62
# ╠═50f7070d-9f6f-4025-9be3-13812c3000eb
# ╟─fa3b6d6a-a55d-4097-8ad2-7dafb5f01d8c
# ╟─dabb4656-4aed-4168-8659-c0472528c41d
# ╟─0367be06-4185-4add-a04b-f696c5a43638
# ╟─c79e7651-975b-407c-8c8c-d0c5653ec570
# ╟─3fed55d7-dbed-4fc9-8410-2633d5200db6
# ╟─663180d2-8fe1-4996-a194-d38120ae05fd
# ╟─39884e8a-bc83-4ff4-85bc-cfaccb4674f2
# ╟─c24e7391-a187-4a7e-aab7-93027f7db965
# ╠═12734842-d66f-4bfc-a9ad-01adeb2450e0
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002

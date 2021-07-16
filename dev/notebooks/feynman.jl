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

# ╔═╡ e81de385-0070-49a9-a889-8fcf9d9e2951
using Plots; gr();

# ╔═╡ db9a97b1-f76d-4f51-96c6-0159469c5adb
using NiLang

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
	md"# Feynman's Lectures on computing, Chap. 5

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
1. Irreversible computing requires dissipating heat to the environment
    * The relation between reversibility and energy cost
        * Compressing boxes
        * An eingine driven by information
    * The relation between computing speed and energy cost
2. several reversible computing models
    * Copy machine
        * Magnetic dipole
        * DNA copying is a type of copy machine
    * General reversible computing
        * Billiard ball model
"

# ╔═╡ f9675365-36aa-430c-b747-3bc4f602e6fb
md"## Information erasure requires dissipating heat to the environment!"

# ╔═╡ 94c5eaa1-432c-4553-829e-f78d97f3c0ca
md"""
* *Information*: uncertainty of system, can be quantified by entropy,
* *Information erasure*: decrease of information entropy,
* *Knowledge*: the complement of information, the more knowledge, the less uncertainty.
"""

# ╔═╡ bef6978d-e654-4364-b5eb-e9608cf68464
md"*setup*: a collection of boxes, with same size ``V``, immersed to a heat bath of temperature of ``T``. Each having a jiggling particle inside."

# ╔═╡ 046f7559-4af9-4982-b5c3-335add0911d7
html"""
<div align=center><img src="https://user-images.githubusercontent.com/6257240/122632611-ef6ad480-d0a1-11eb-976c-a3e7c5dfdb9a.png" width=500/></div>
"""

# ╔═╡ 0a039bfa-571e-4fad-b73c-1324d08777fc
html"""
<div align=center><img src="https://user-images.githubusercontent.com/6257240/122682827-b168d000-d1c9-11eb-930c-0ff13a2bf631.png" width=300/></div>
"""

# ╔═╡ bf7abacc-5b0a-4623-b2c5-af60183ad4b0
md"""
A piston attached to each box.
"""

# ╔═╡ 3f1e4d7a-32a7-4c7e-92dd-465bac925e63
md"""
*goal*: Compress the boxes from size $V$ to size $V/2$, the process is isothermal.
"""

# ╔═╡ 95a21058-0b07-4859-af68-8ca5b48b2a77
md"## Case 1: Ideal Gas"

# ╔═╡ f68bcfb6-97ce-48d1-b0b8-e8466d4ac879
md"""
Case 1: we know nothing about the system. The gas does work
```math
\begin{align}
&pV = N \underbrace{k}_{\substack{\text{Boltzman constant}\\\sim 1.38 × 10-23 m^2 {\rm kg} s^{-2} K^{-1}}} T\\
&W_{\rm gas} = \int_{V}^{V/2} p dV = -NkT\log 2
\end{align}
```
"""

# ╔═╡ 2fe7c298-4c5d-464c-980b-6cd9a537ac1e
let
	img = TikzPicture(L"""
	\draw (-1.05, -1.05) rectangle (1.1,1.1);
	\foreach[evaluate={\a=rand; \b=rand;\c=rand;\d=rand;}] \x in {1,...,20}{
		\fill (\a, \b) circle [radius=0.05];
		\draw[->,thick] (\a, \b) -- (\a+\c*0.2, \b+\d*0.2);
	}
\node[draw, single arrow, minimum height=10mm, minimum width=3mm,
              single arrow head extend=2mm, rotate=90] at (0.0,1.3) {presure};
\node[draw, single arrow, minimum height=10mm, minimum width=3mm,
              single arrow head extend=2mm, rotate=-90] at (0.0,-1.3) {presure};
\node[draw, single arrow, minimum height=10mm, minimum width=3mm,
              single arrow head extend=2mm, rotate=0] at (1.3,0.0) {presure};
\node[draw, single arrow, minimum height=10mm, minimum width=3mm,
              single arrow head extend=2mm, rotate=180] at (-1.3,0.0) {presure};
"""; options="scale=2.0", preamble=raw"\usetikzlibrary{shapes.arrows}")
	leftright(img, md"the microscopic explaination of the presure
		
``kT \sim \text{average kinetic energy}``")
end

# ╔═╡ 49dab78a-7bd9-4faa-8a30-9af8a96e0c5b
md"## Case 2: With prior knowledge"

# ╔═╡ 3d4ba750-8d62-48ac-bf96-691397689ddc
md"""
Case 2: we know one bit knowledge about each box:
* 1: the atom is in the right half
* 0: the atom is in the left half

```math
W_{\rm gas} = 0
```
"""

# ╔═╡ 7aa7b0ee-beeb-4a3e-abf1-aa71e916f4cd
md"""
## Compare
Case 1:

* erase information (the left-right information of the atom),
* consumes energy

Case 2:
* do not erase information,
* does not consume energy
"""

# ╔═╡ f4cb9212-181f-4338-b858-1d99c7f415e9
md"Erasing each bit information comes along with $kT \log 2$ heat dissipation!!"

# ╔═╡ c1bbaec8-4fb9-4ab8-a30d-06a286597de0
md"""
## Maxwell's demon

The *Second Law of Thermodynamics* states that the state of entropy of the entire universe, as an isolated system, will always increase over time. The second law also states that the changes in the entropy in the universe can never be negative.

![](https://user-images.githubusercontent.com/6257240/124372430-b14fe200-dc57-11eb-9e8d-75385e2c621b.png)
"""

# ╔═╡ 6d7a07ff-be1b-4902-8a6d-7d9257c1157f
md"""
Before observing: (s, t), number of possible configurations ``2^{|s|+|t|}``

After observing: (s, t=s), number of possible configurations ``2^{|s|}``
"""

# ╔═╡ 0577d67f-648f-407c-8abf-507d086445bd
md"""
## Proving from the quantum setup
"""

# ╔═╡ eb10e436-bcce-4d81-891e-15158219fe80
md"Reeb (2014)"

# ╔═╡ 7c5a30fd-95f9-4bb8-b34f-b10b0f2a27f2
md"""
1. the process involves a “system” ``S`` and a “reservoir” ``R``, both described by Hilbert spaces,
2. the reservoir ``R`` is initially in a thermal state, ``\rho_R = e^{−\beta H}/{\rm tr}[e^{−\beta H}]`` , where H is a Hermitian operator on ``R`` (“Hamiltonian”) and ``\beta \in [−\infty, +\infty]`` is the “inverse temperature”,
3. the system ``S`` and the reservoir ``R`` are initially uncorrelated, ``\rho_{SR} = \rho_S \otimes \rho_R``, 
4. the process itself proceeds by unitary evolution, ``\rho_{SR}'=U\rho_{SR}U^\dagger``.
"""

# ╔═╡ abee1bee-ed01-4b05-a848-3aeb695a24ba
md"""
![](https://user-images.githubusercontent.com/6257240/124468410-0e868900-dd67-11eb-91b4-b9ab92f21152.png)
"""

# ╔═╡ b05538cc-de01-4b1e-a602-feb780cddf4a
md"""
Main result: ``\Delta > \Delta S``, because
```math
\begin{align}
[S(\rho_S') - S(\rho_S)] + [S(\rho_R')-S(\rho_R)] &=[S(\rho_S') + S(\rho_R')-S(\rho_{SR})] \ldots (3)\\
&=[S(\rho_S') + S(\rho_R')-S(\rho_{SR}')] \ldots (4)\\
&=I(S': R') \geq 0
\end{align}
```
"""

# ╔═╡ 42c398ab-bb45-423f-b030-404e7582df5a
md"""
## An information driven car
"""

# ╔═╡ 48081dd4-2bf4-43a1-899c-0303b4fcedd3
md"""
![](https://user-images.githubusercontent.com/6257240/124372207-2bcc3200-dc57-11eb-840e-1bf2c85abf9b.png)"""

# ╔═╡ c6ef8479-639b-45c1-9b48-a5d2c233d3b8
md"1. set up the initial state to ``0``, contacting with a heat bath of temperature ``T``,
2. place a piston at the half way of the box
3. the environment warm up the box
4. the particle isothermally push the piston outwards"

# ╔═╡ 6f01cdc2-6ce9-41da-b279-b047c9779405
md"## An example of reversible computing: Copy machine"

# ╔═╡ 876ad6cf-84c1-4e34-89de-6f9273ba3479
md"*setup*: A copier (state known) and a model (state unknown),
both being double well potentials. In figure
* ``x`` axis is the parameter,
* ``y`` axis is the energy.
"

# ╔═╡ 9ca8912d-5fc5-4066-adb8-ad02f75c2cbe
md"``0`` state, left well

``1`` state, right well"

# ╔═╡ 9aab5751-e9e0-46c0-8e66-4b98258fed08
md"""
![](https://user-images.githubusercontent.com/6257240/124372454-cc225680-dc57-11eb-8526-ed397ce10583.png)
"""

# ╔═╡ a29af398-ff22-44cb-a5aa-0b0409312be9
md"There is a tilt force when we make two double wells close"

# ╔═╡ c3db622f-e9ff-4d99-afb6-9db65c6cae7a
md"
*goal*: set the state of copier to the same state as the model.
"

# ╔═╡ 12cbf4b7-9b55-423c-bf59-5cb18e167afd
md"*procedure*"

# ╔═╡ 89a5ff44-1b04-4bd8-a40a-83382a027fb3
md"Step 1: lowering the copier's potential barrier."

# ╔═╡ 51e7b853-8640-4415-a9a4-8c0e06ad916a
md"Step 2: bring the model close to copier (above illustration)."

# ╔═╡ a8fe838e-727d-4068-887d-17b1bf99f90b
md"Step 3: raise the copier's potential barrier."

# ╔═╡ c7cd75cb-4c64-4704-b839-c5a556f89be7
md"Step 4: take the model away"

# ╔═╡ ec14fba6-0cb9-483f-b3ea-cc4c5e83c965
md"## Magnetic dipole

[ref](https://en.wikipedia.org/wiki/Magnetic_dipole)"

# ╔═╡ f1abc5c1-2c34-422a-86c4-5ad8e7df8b7e
md"""
*setup*: two magnetic dipoles pointing to the same direction.
"""

# ╔═╡ 8ad4e7c0-c496-4d29-ac09-e6525b1b4c0f
md"""
![](https://user-images.githubusercontent.com/6257240/124498319-23c0df00-dd8a-11eb-9fca-51a87fde6ec0.png)
"""

# ╔═╡ 757e2d78-c5ee-4b40-bfd6-1b39af338d9d
md"""
```math
\text{potential energy} \approx \sin^2 \phi
```
"""

# ╔═╡ cbea35c7-c3c8-48e7-bb47-d5e193aee2c4
md"""
state ``0``: ← ←

state ``1``: → →
"""

# ╔═╡ 81013954-1c48-4c05-82c9-49b4bfafda95
let
	x = 0:1000
	y = map(x->sin(x/1000*2π)^2, x)
	Plots.plot(x./1000, y, xlabel="ϕ/2π", ylabel="potential energy")
end

# ╔═╡ 1924eff7-1423-4e90-8005-43113d9deb3d
md"""
Step 1: Introduce a vertical magnetic field ``B``, we have
```math
\text{potential energy (magnetic field)} = -B \sin \phi
```
"""

# ╔═╡ c7edbc15-cd59-45fd-a0dc-c48aadb1c096
md"B = $(@bind B Slider(0:0.01:2; show_value=true))"

# ╔═╡ bf2c9da7-8c45-409f-82a3-979cd63ea993
let
	x = 0:1000
	y = map(x) do x
		ϕ = x/1000*2π
		sin(ϕ)^2 - B * sin(ϕ)
	end
	Plots.plot(x./1000, y, xlabel="ϕ/2π", ylabel="potential energy")
end

# ╔═╡ 1c32a491-ac85-4132-82fd-9b846a8485df
md"""
copier state is ``\uparrow \uparrow``
"""

# ╔═╡ 6cbf202f-34e4-42b6-a7a7-5d766bfdfc37
md"""
Step 2: bring the model  (assume it is in state 1) close to the copier

```math
\text{potential energy (model)} = -b \cos \phi
```
"""

# ╔═╡ d40b318f-bff2-4d0b-b2a6-d00933ac7567
md"b = $(@bind b Slider(-0.5:0.01:0.5; default=0.0,show_value=true))"

# ╔═╡ 54f53a7b-74e8-433b-94fd-9fa7192dfca5
let
	x = 0:1000
	y = map(x) do x
		ϕ = x/1000*2π
		sin(ϕ)^2 - B * sin(ϕ) - b*cos(ϕ)
	end
	Plots.plot(x./1000, y, xlabel="ϕ/2π", ylabel="potential energy")
end

# ╔═╡ b0dcad96-e439-4e09-9e92-8cad7ede79af
md"""
# Last time
* Erase information -> make the system into a more certain state -> a decrease of entropy in the system -> requires dissipating heat to the heat bath (Landauer's principle), and this can be proved regorously from the quantum perspective.
* Introduce a type of reversible computing model: copy machine
    * Magnetic dipole
    * Protein synthesis
"""

# ╔═╡ 0a15a2cf-2e7a-4bd7-ac78-0803fc3d5c73
md"""
# This time
* The relation between energy and speed in reversible computing.
* General reversible gates and reversible programming,
    * The billiard ball model
    * Reversible control flows
* Several reversible computing architectures
"""

# ╔═╡ ffbc5616-d2d9-4ce4-996f-d1a743bb89b3
md"## Protein synthesis （Brownian computer）"

# ╔═╡ 2602d857-4a21-478a-97a2-58a177666f52
md"""
*setup*: we only consider the first stage of protein synthesis, copying information from DNA to m-RNA. A DNA strand is immersed to a biological soup with lots of triphostrates such as ATP, CTP, GTP and UTP.

A DNA strand is made up of alternating phosphate and pentose
sugar groups. To each sugar group is attached one of four bases, A (adenine),
T (thymine), C (cytosine) and G (guanine)
"""

# ╔═╡ cf27e340-578a-440d-8d4a-e5a2277d5205
md"""
![](https://user-images.githubusercontent.com/6257240/122641081-f957fc00-d0d0-11eb-9c7b-180e11f9bc33.png)
"""

# ╔═╡ aa53fd68-5acd-488d-a096-5ce39759f481
md"lowering potential: enzyme"

# ╔═╡ cb9a9ef0-c0dc-487c-8008-0f73f9910ef8
md"""
key point: chemecal reaction is reversible, the direction to evolve depends on the relative concentrations of pyrophosphates and triphosphates
in the soup
"""

# ╔═╡ f7e0478d-1839-4684-9265-ee990fe9da45
md"## Speed and energy in a Brownian computer"

# ╔═╡ 751e32d6-2582-4b1d-9558-124b1ef54f81
md"""
![](https://user-images.githubusercontent.com/6257240/124506870-8c17bc80-dd9a-11eb-9144-4116cf00f1c2.png)
"""

# ╔═╡ 5f18987d-a69e-4db9-96d3-426ed298d9b8
md"""
Thermal fluctuation helps overcome the barrier
```math
\text{forward rate} = C X e^{-(A-E_1)/kT}
```
```math
\text{backward rate} = C X e^{-(A-E_2)/kT}
```

``C`` is a factor that carries information about the thermal fluctuations in the
environment

``X`` is a factor depends on a variety of molecular properties of the particular
substance
"""

# ╔═╡ 66495c77-3bbc-4731-b9e1-db11bbc24283
md"*analysis*:
The rate of forward/backward computing is 
```math
r = \frac{\text{forward rate}}{\text{backward rate}} = e^{(E_1-E_2)/kT}
```
The minimum free energy/step we need to pay is ``kT \log r = (E_1-E_2)``
"

# ╔═╡ 84b867a3-804e-4e7e-a56c-0ffc1f4e6683
md"""
## Speed v.s. energy efficiency - the entropy perspective
"""

# ╔═╡ 4642d311-ef0b-4c29-901d-b5398a3ca7b6
md"""
![](https://user-images.githubusercontent.com/6257240/124663469-200b8600-de78-11eb-8501-8ce97ea140c5.png)
"""

# ╔═╡ 96c3d50b-8a79-4de0-b7e0-c63c3b769b74
md"The ratio of the forward rate and backward rate"

# ╔═╡ 066aa825-81e4-404d-bf5a-6a9431969702
md"``r = n_2/n_1`` (determines the speed)"

# ╔═╡ 6e99ed64-a896-450e-8bab-845e0fe971ae
md"``kT\log r = \underbrace{(S_2 - S_1) T}_{\text{the cost of free energy}}``"

# ╔═╡ 7f803113-653a-4dfd-93f0-83babb253b32
md"$F = E - TS$"

# ╔═╡ 7eb29d49-05f5-47e9-b4f5-4f31c5cd37ce
md"## A more concrete example about the energy efficiency and speed"

# ╔═╡ aefdec07-dcef-4e00-bcb0-4747250cdd9b
md"Charging up the capacitor to store signal energy in adiabatic CMOS."

# ╔═╡ cbe4abaf-46f9-4726-97ae-cf3c378abaaf
md"""
![](https://user-images.githubusercontent.com/6257240/122668453-287c7500-d186-11eb-962f-cc478be1dafe.png)
"""

# ╔═╡ 4f0c81f5-ce5f-4f73-a528-9feff4a7fc14
md"Case 1: do it fast
```math
E = CU^2
```
"

# ╔═╡ 8249b820-8fb1-45d4-a95c-9c81e62e8216
let
	x = 0:1000
	y = map(x->abs(x-500)<250 ? 1.0 : 0.0, x)
	Plots.plot(x./1000, y, xlabel="time", ylabel="voltage")
end

# ╔═╡ 6503b377-b2d5-48be-90a4-97947afb4e5f
md"Case 1: do it slow
```math
E = \frac{CU^2}{2}
```
"

# ╔═╡ 53a571dd-cac7-432a-869c-b93a8fe05e17
let
	x = 0:1000
	y = map(x->abs(x-500)<200 ? 1.0 : (abs(x-500) < 400 ? 1/200 * (x < 500 ? abs(x-100) : abs(900-x)) : 0.0), x)
	Plots.plot(x./1000, y, xlabel="time", ylabel="voltage")
end

# ╔═╡ 2e5c7f59-dd35-4846-815a-b92eabeee089
md"Conclusion: fast ramping dissipates heat to the environment."

# ╔═╡ 7b326477-43b6-4a6e-8862-12e8b70e1ad9
md"## Universal reversible gate set"

# ╔═╡ 47cd7560-a29e-4b55-bef5-28daa1cdb834
md"Toffoli gate is universal"

# ╔═╡ 59ab4431-ea4d-4707-9a42-d50eafa40b56
md"truth table ``(A, B, C) \mapsto (A, B, C')``

|  A  |  B  |  C  |  C' |
| --- | --- | --- | --- |
| 0 | 0 | 0 | 0 |
| 0 | 0 | 1 | 1 |
| 0 | 1 | 0 | 0 |
| 0 | 1 | 1 | 1 |
| 1 | 0 | 0 | 0 |
| 1 | 0 | 1 | 1 |
| 1 | 1 | 0 | 1 |
| 1 | 1 | 1 | 0 |
"

# ╔═╡ 3630b412-beeb-455a-a4b8-1e1d50860266
md"
```julia
if A && B
	C = ¬C
end
```
"

# ╔═╡ ba6347d6-4ad0-403b-824f-dcf290a7c002
md"proved by constructing a NAND gate (a classical univeral gate)"

# ╔═╡ a2f4975d-eeee-4a2d-97dd-dd0cfd29d665
TikzPicture(L"""
\draw (0,0) rectangle (2,3);
\draw (-0.5,0.5) -- (0.0, 0.5);
\node at (-0.8, 0.5) {1};
\draw (-0.5,1.5) -- (0.0, 1.5);
\node at (-0.8, 1.5) {B};
\draw (-0.5,2.5) -- (0.0, 2.5);
\node at (-0.8, 2.5) {A};
\draw (2.5,0.5) -- (2.0, 0.5);
\node at (3.1, 0.5) {$\overline{A\land B}$};
\draw (2.5,1.5) -- (2.0, 1.5);
\node at (2.8, 1.5) {B};
\draw (2.5,2.5) -- (2.0, 2.5);
\node at (2.8, 2.5) {A};
\node at (1.0, 1.5) {Toffoli};
""")

# ╔═╡ 32d411e9-b01d-4ad2-b4aa-2f091034e6c0
md"Fredkin gate is universal"

# ╔═╡ e5b83421-dd94-43ad-84eb-ca558bff6a2d
md"truth table ``(A, B, C) \mapsto (A, B', C')``

|  A  |  B  |  C  |  B' |  C' |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 | 0 |
| 0 | 0 | 1 | 0 | 1 |
| 0 | 1 | 0 | 1 | 0 |
| 0 | 1 | 1 | 1 | 1 |
| 1 | 0 | 0 | 0 | 0 |
| 1 | 0 | 1 | 1 | 0 |
| 1 | 1 | 0 | 0 | 1 |
| 1 | 1 | 1 | 1 | 1 |
"

# ╔═╡ 118642ad-1aad-4f91-8da0-55a417b67750
md"
```julia
if A
	B, C = C, B
end
```
"

# ╔═╡ 45171ecc-9d34-4ab6-a00b-ec9c9afc33f8
md"prove by constructing an AND gate and NOT gate"

# ╔═╡ 6cd60f7d-d7ce-4189-a2dd-e47ce6825741
let
	img1 = TikzPicture(L"""
\draw (0,0) rectangle (2,3);
\draw (-0.5,0.5) -- (0.0, 0.5);
\node at (-0.8, 0.5) {$C$};
\draw (-0.5,1.5) -- (0.0, 1.5);
\node at (-0.8, 1.5) {$0$};
\draw (-0.5,2.5) -- (0.0, 2.5);
\node at (-0.8, 2.5) {$A$};
\draw (2.5,0.5) -- (2.0, 0.5);
\node at (3.1, 0.5) {$\overline{A}\land C$};
\draw (2.5,1.5) -- (2.0, 1.5);
\node at (3.1, 1.5) {$A\land C$};
\draw (2.5,2.5) -- (2.0, 2.5);
\node at (2.8, 2.5) {$A$};
\node at (1.0, 1.5) {Fredkin};
""")
	img2 = TikzPicture(L"""
\draw (0,0) rectangle (2,3);
\draw (-0.5,0.5) -- (0.0, 0.5);
\node at (-0.8, 0.5) {$1$};
\draw (-0.5,1.5) -- (0.0, 1.5);
\node at (-0.8, 1.5) {$0$};
\draw (-0.5,2.5) -- (0.0, 2.5);
\node at (-0.8, 2.5) {$A$};
\draw (2.5,0.5) -- (2.0, 0.5);
\node at (2.8, 0.5) {$\overline{A}$};
\draw (2.5,1.5) -- (2.0, 1.5);
\node at (2.8, 1.5) {$A$};
\draw (2.5,2.5) -- (2.0, 2.5);
\node at (2.8, 2.5) {$A$};
\node at (1.0, 1.5) {Fredkin};
""")
	leftright(img1, img2)
end

# ╔═╡ ff3fc929-f448-41be-8f60-65de33dff36a
md"""
## Billiard Ball Computer
"""

# ╔═╡ 5ec2649e-9988-4f38-896a-64ef6ed91d82
md"*setup*: Billiard balls in 2D space"

# ╔═╡ aa8475c3-c68b-4200-8634-ace33f525417
md"The collision Gate"

# ╔═╡ aa22f905-b69d-405e-b09a-a765d60f6079
md"""
![](https://user-images.githubusercontent.com/6257240/124666172-a2497980-de7b-11eb-9221-378f0453d41d.png)
"""

# ╔═╡ 2dcbaac0-2fad-4292-ad31-8188a60876da
md"Four redirection gates"

# ╔═╡ 6bad4f5f-806f-480a-ae16-2582761ce5e3
md"""
![](https://user-images.githubusercontent.com/6257240/124666268-cc02a080-de7b-11eb-9017-c62d9d251521.png)
"""

# ╔═╡ e200dde3-9033-45b5-bfe0-2d03753b2c11
md"*Define some other useful gates*:"

# ╔═╡ d7a4b342-ef0a-44f9-b88d-bbb04483e8b3
let
	img1 = md"""
![](https://user-images.githubusercontent.com/6257240/124670709-62d25b80-de82-11eb-9c1b-25377e16c43a.png)
"""
	img2 = md"""
![](https://user-images.githubusercontent.com/6257240/124670724-6960d300-de82-11eb-8ebb-4747e0d43fff.png)
"""
	leftright(img1, img2)
end

# ╔═╡ 908f19a2-6d32-4776-95eb-b249a8155ddc
md"""prove by mapping it to a Fredkin gate"""

# ╔═╡ 05fc1fae-b378-4c39-b060-74ca635745ec
md"""![](https://user-images.githubusercontent.com/6257240/124670762-78478580-de82-11eb-8061-409e0db1388c.png)"""

# ╔═╡ d0573bf9-0fd6-4512-bc13-17aa23a3265b
md"""
## General reversible computing
"""

# ╔═╡ d8998d5f-65b2-4850-aef9-f19ecc192eca
md"""*Problem 5.4*: A related problem concerns how to get "if' clauses to work. What
if, after having followed an "if... then ... " command, the machine starts to
reverse? How can the machine get back to the original condition that dictated
which way the "if' branched? Of course, a set of initial conditions can result in
a single "if' output ("if x = 2, 3, 4 or 6.159 let F= d"), so this condition may not be uniquely specified. Here is a nice way to analyze things. Simply bring in a
new variable at each branch, and assign a unique value to this variable for each
choice at a branch point. You might like to work this through in detail."""

# ╔═╡ 6b4c180c-9a12-4e3d-9336-1431e7c5875a
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

# ╔═╡ 0eb66cc9-93c0-4f07-b31b-a9bf9000260e
md"Reversible branching statement"

# ╔═╡ 85bf9f92-30f1-4e05-8d07-d8e481f20ccb
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

# ╔═╡ fbba0a91-9f48-4d91-90e7-f6a7df3227f9
md"## Bennett's compute copy uncompute scheme"

# ╔═╡ 7fc81b9f-73ed-4780-9204-ddf39467e58f
md"*setup*: we have a long linear program. We use the reversible embeded domain specific language in Julia"

# ╔═╡ d08ae188-937f-474b-92d7-cb8eeda063fe
html"""<img src="https://github.com/GiggleLiu/NiLang.jl/raw/master/docs/src/asset/logo3.png" width=200/>"""

# ╔═╡ 7ee8cfc9-26b2-4fe4-8263-1f4d2f7c276d
md"Initially written by GiggleLiu and Taine Zhao (The author of MLStyle)"

# ╔═╡ 1669f5e3-efe1-4b79-a2b6-11ed7476a2a1
md"the program of finding the maximum number"

# ╔═╡ 5000f4c3-5416-4e53-88ae-e30d8d09827e
@i function i_find_maximum_v1(s₂, s₃, s₄, x₁, x₂, x₃, x₄) where T
	s₂ += max(x₁, x₂) # step 1
	s₃ += max(s₂, x₃) # step 2
	s₄ += max(s₃, x₄) # step 3
end

# ╔═╡ 2413c061-89de-403f-8011-e458f5a9859d
i_find_maximum_v1(0, 0, 0, 3, 2, 8, 1)

# ╔═╡ d4704779-9261-478b-bbf6-551220783e12
md"the basic building block of compute-copy-uncompute"

# ╔═╡ 4be065b5-0841-4d54-b9ab-d6770d4d9d94
@i function i_find_maximum_v2(s₄, x₁, x₂, x₃, x₄) where T
	# compute
	s₂ ← 0  # variable on the working tape
	s₃ ← 0
	s₂ += max(x₁, x₂) # step 1
	s₃ += max(s₂, x₃) # step 2
	
	# copy
	s₄ += max(s₃, x₄) # step 3
	
	# uncompute
	s₃ -= max(s₂, x₃) # step 4
	s₂ -= max(x₁, x₂) # step 5
	s₂ → 0
	s₃ → 0
end

# ╔═╡ e850e53d-cf61-4fc7-9cb3-e318ae957f0b
i_find_maximum_v2(0, 3, 2, 8, 1)

# ╔═╡ a267ea5f-8bd5-4ee0-9c8d-47e2d3b81692
TikzPicture(L"""
\def\r{0.15};
\foreach \x in {1,4}{
	\fill[fill=black] (\x, 0) circle [radius=\r];
	\node[white] at (\x, 0) {$s_{\x}$};
}
\foreach \x in {2,3}{
	\draw (\x, 0) circle [radius=\r];
	\node[black] at (\x, 0) {$s_{\x}$};
}
\fill[fill=white] (5.5, 0) circle [radius=\r];
\foreach \x in {1,...,3}{
	\draw [black, thick, ->] (\x+\r, \r) .. controls (\x+0.5, 0.3) .. (\x+1-\r, \r);
	\node at (\x+0.5, 0.4) {\x};
	}
\foreach[evaluate={\y=int(6-\x)}] \x in {1,...,2}{
	\draw [red, thick, <-] (\x+\r, -\r) .. controls (\x+0.5, -0.3) .. (\x+1-\r, -\r);
	\node at (\x+0.5, -0.4) {\y};
	}
"""
, options="scale=1.8", preamble="")

# ╔═╡ c02520a3-3375-4d83-a0dc-1aeac2aa7d5f
md"Recursively apply Bennett's time space tradeoff scheme"

# ╔═╡ 2e18fc92-4185-493b-9ce8-cca63dad7d2d
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

# ╔═╡ acc7b185-e4df-4aca-aa42-554215065384
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

# ╔═╡ 00c9e973-7e06-4483-bf4c-be7374707118
md"
* Space complexity: ``O(S \log T)``
* Time complexity: ``O(T^{1+\epsilon})``"

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
md"## More


![](https://user-images.githubusercontent.com/6257240/123520518-22ebc700-d67f-11eb-8af1-a452605cc1d8.png)

*Youtube*: Michael P. Frank: Fundamental Physics of Reversible Computing — An Introduction, Part 1
"

# ╔═╡ 74017e78-0f02-41bb-a160-5f2d26c18268
md"""
![](https://user-images.githubusercontent.com/6257240/125467165-d3ec7c24-18cb-48d6-99b6-708326789bf9.png)

	
Kenichi Morita, How can we construct reversible Turing machines in a very simple reversible cellular automaton? Video can be found in this conference page: [https://reversible-computation-2021.github.io/program/](https://reversible-computation-2021.github.io/program/)
"""

# ╔═╡ 0b3735c2-695c-4225-843e-16ca17aac0eb
md"""## Take home message

1. Irreversible computing -> information erasure -> disspate heat (``kT\log 2`` per bit)
2. Reversible computing -> requires operations being adiabatic -> slow
2. Reversible programming suffers from **polynomial time overhead and logarithmic space overhead** when differentiating a irreversible linear program
3. Brownian computer
    * mRNA copy
    * Magnetic dopile
4. General reversible computer
    * Billiard ball model
    * Reversible cellular automata
    * Adiabatic CMOS

6. **How to find this notebook?** In NiLang's Github repo, file: `notebooks/feynman.jl`
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

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Compose = "a81c6b42-2e10-5240-aca2-a61377ecd94b"
NiLang = "ab4ef3a6-0b42-11ea-31f6-e34652774712"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Revise = "295af30f-e4ad-537b-8983-00126c2a3abe"
TikzPictures = "37f6aa50-8035-52d0-81c2-5a1d08754b2d"
Viznet = "52a3aca4-6234-47fd-b74a-806bdf78ede9"

[compat]
Compose = "~0.9.2"
NiLang = "~0.9.1"
Plots = "~1.18.0"
PlutoUI = "~0.7.9"
Revise = "~3.1.17"
TikzPictures = "~3.3.3"
Viznet = "~0.3.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c3598e525718abcc440f69cc6d5f60dda0a1b61e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.6+5"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "e2f47f6d8337369411569fd45ae5753ca10394c6"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.0+6"

[[CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "8ad457cfeb0bca98732c97958ef81000a543e73e"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.0.5"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random", "StaticArrays"]
git-tree-sha1 = "c8fd01e4b736013bc61b704871d20503b33ea402"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.12.1"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "32a2b8af383f11cbb65803883837a149d10dfe8a"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.10.12"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

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

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[DataAPI]]
git-tree-sha1 = "ee400abb2298bd13bfc3df1c412ed228061a2385"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.7.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4437b64df1e0adccc3e5d1adbc3ac741095e4677"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.9"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

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

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "92d8f9f208637e8d2d28c664051a00569c01493d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.1.5+1"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "LibVPX_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "3cc57ad0a213808473eafef4845a74766242e05f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.3.1+4"

[[FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "35895cf184ceaab11fd778b4590144034a167a2f"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.1+14"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "cbd58c9deb1d304f5a245a0b7eb841a2560cfec6"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.1+5"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "dba1e8614e98949abfa60480b13653813d8f0157"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+0"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "b83e3125048a9c3158cbb7ca423790c7b1b57bea"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.57.5"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "e14907859a1d3aee73a019e7b3c98e9e7b8b5b3e"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.57.3+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "15ff9a14b9e1218958d3530cc288cf31465d9ae2"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.3.13"

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

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "c6a1fff2fd4b1da29d3dccaffb1e1001244d844e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.12"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

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

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

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

[[LibVPX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "12ee7e23fa4d18361e7c2cde8f8337d4c3101bc7"
uuid = "dd192d2f-8180-539f-9fb4-cc70b1dcf69a"
version = "1.10.0+0"

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

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

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

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "4ea90bd5d3985ae1f9a908bd4500ae88921c5ce7"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.0"

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

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenJpeg_jll]]
deps = ["Libdl", "Libtiff_jll", "LittleCMS_jll", "Pkg", "libpng_jll"]
git-tree-sha1 = "e330ffff1c6a593fa44cc40c29900bee82026406"
uuid = "643b3616-a352-519d-856d-80112ee9badc"
version = "2.3.1+0"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

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

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "501c20a63a34ac1d015d5304da0e645f42d91c9f"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.11"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "9f126950870ef24ce75cdd841f4b7cf34affc6d2"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.18.0"

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

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "b3fb709f3c97bfc6e948be68beeecb55a0b340ae"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.1"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "2a7a2469ed5d94a98dea0e85c46fa653d76be0cd"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.3.4"

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

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "2ec1962eba973f383239da22e75218565c390a96"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.0"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "896d55218776ab8f23fb7b222a5a4a946d4aafc2"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.5"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2f6792d523d7448bbe2fec99eca9218f06cc746d"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.8"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "000e168f5cc9aded17b6999a560b7c11dda69095"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.0"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "8ed4a3ea724dac32670b062be3ef1c1de6773ae8"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.4.4"

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

[[TupleTools]]
git-tree-sha1 = "3c712976c47707ff893cf6ba4354aa14db1d8938"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.3.0"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

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

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

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

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

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

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

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

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

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

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "acc685bcf777b2202a904cdcb49ad34c2fa1880c"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.14.0+4"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7a5780a0d9c6864184b3a2eeeb833a0c871f00ab"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "0.1.6+4"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d713c1ce4deac133e3334ee12f4adff07f81778f"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2020.7.14+2"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "487da2f8f2f0c8ee0e83f39d13037d6bbf0a45ab"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.0.0+3"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─e20e2d2e-4b28-4e32-8d80-ce029928a094
# ╟─f3e235e7-76b9-4c39-bc70-038539838ff4
# ╟─8308df59-3faa-4abf-8f05-119bbae48f64
# ╟─a3532a83-9fd3-4d24-b1bb-b52457317e51
# ╟─15657e4b-848e-43ad-a99f-37143d11705e
# ╟─f9675365-36aa-430c-b747-3bc4f602e6fb
# ╟─94c5eaa1-432c-4553-829e-f78d97f3c0ca
# ╟─bef6978d-e654-4364-b5eb-e9608cf68464
# ╟─046f7559-4af9-4982-b5c3-335add0911d7
# ╟─0a039bfa-571e-4fad-b73c-1324d08777fc
# ╟─bf7abacc-5b0a-4623-b2c5-af60183ad4b0
# ╟─3f1e4d7a-32a7-4c7e-92dd-465bac925e63
# ╟─95a21058-0b07-4859-af68-8ca5b48b2a77
# ╟─f68bcfb6-97ce-48d1-b0b8-e8466d4ac879
# ╟─2fe7c298-4c5d-464c-980b-6cd9a537ac1e
# ╟─49dab78a-7bd9-4faa-8a30-9af8a96e0c5b
# ╟─3d4ba750-8d62-48ac-bf96-691397689ddc
# ╟─7aa7b0ee-beeb-4a3e-abf1-aa71e916f4cd
# ╟─f4cb9212-181f-4338-b858-1d99c7f415e9
# ╟─c1bbaec8-4fb9-4ab8-a30d-06a286597de0
# ╟─6d7a07ff-be1b-4902-8a6d-7d9257c1157f
# ╟─0577d67f-648f-407c-8abf-507d086445bd
# ╟─eb10e436-bcce-4d81-891e-15158219fe80
# ╟─7c5a30fd-95f9-4bb8-b34f-b10b0f2a27f2
# ╟─abee1bee-ed01-4b05-a848-3aeb695a24ba
# ╟─b05538cc-de01-4b1e-a602-feb780cddf4a
# ╟─42c398ab-bb45-423f-b030-404e7582df5a
# ╟─48081dd4-2bf4-43a1-899c-0303b4fcedd3
# ╟─c6ef8479-639b-45c1-9b48-a5d2c233d3b8
# ╟─6f01cdc2-6ce9-41da-b279-b047c9779405
# ╟─876ad6cf-84c1-4e34-89de-6f9273ba3479
# ╟─9ca8912d-5fc5-4066-adb8-ad02f75c2cbe
# ╟─9aab5751-e9e0-46c0-8e66-4b98258fed08
# ╟─a29af398-ff22-44cb-a5aa-0b0409312be9
# ╟─c3db622f-e9ff-4d99-afb6-9db65c6cae7a
# ╟─12cbf4b7-9b55-423c-bf59-5cb18e167afd
# ╟─89a5ff44-1b04-4bd8-a40a-83382a027fb3
# ╟─51e7b853-8640-4415-a9a4-8c0e06ad916a
# ╟─a8fe838e-727d-4068-887d-17b1bf99f90b
# ╟─c7cd75cb-4c64-4704-b839-c5a556f89be7
# ╟─ec14fba6-0cb9-483f-b3ea-cc4c5e83c965
# ╟─f1abc5c1-2c34-422a-86c4-5ad8e7df8b7e
# ╟─8ad4e7c0-c496-4d29-ac09-e6525b1b4c0f
# ╟─757e2d78-c5ee-4b40-bfd6-1b39af338d9d
# ╟─cbea35c7-c3c8-48e7-bb47-d5e193aee2c4
# ╟─81013954-1c48-4c05-82c9-49b4bfafda95
# ╟─1924eff7-1423-4e90-8005-43113d9deb3d
# ╟─c7edbc15-cd59-45fd-a0dc-c48aadb1c096
# ╟─bf2c9da7-8c45-409f-82a3-979cd63ea993
# ╟─1c32a491-ac85-4132-82fd-9b846a8485df
# ╟─6cbf202f-34e4-42b6-a7a7-5d766bfdfc37
# ╟─d40b318f-bff2-4d0b-b2a6-d00933ac7567
# ╟─54f53a7b-74e8-433b-94fd-9fa7192dfca5
# ╟─b0dcad96-e439-4e09-9e92-8cad7ede79af
# ╟─0a15a2cf-2e7a-4bd7-ac78-0803fc3d5c73
# ╟─ffbc5616-d2d9-4ce4-996f-d1a743bb89b3
# ╟─2602d857-4a21-478a-97a2-58a177666f52
# ╟─cf27e340-578a-440d-8d4a-e5a2277d5205
# ╟─aa53fd68-5acd-488d-a096-5ce39759f481
# ╟─cb9a9ef0-c0dc-487c-8008-0f73f9910ef8
# ╟─f7e0478d-1839-4684-9265-ee990fe9da45
# ╟─751e32d6-2582-4b1d-9558-124b1ef54f81
# ╟─5f18987d-a69e-4db9-96d3-426ed298d9b8
# ╟─66495c77-3bbc-4731-b9e1-db11bbc24283
# ╟─84b867a3-804e-4e7e-a56c-0ffc1f4e6683
# ╟─4642d311-ef0b-4c29-901d-b5398a3ca7b6
# ╟─96c3d50b-8a79-4de0-b7e0-c63c3b769b74
# ╟─066aa825-81e4-404d-bf5a-6a9431969702
# ╟─6e99ed64-a896-450e-8bab-845e0fe971ae
# ╟─7f803113-653a-4dfd-93f0-83babb253b32
# ╟─7eb29d49-05f5-47e9-b4f5-4f31c5cd37ce
# ╟─aefdec07-dcef-4e00-bcb0-4747250cdd9b
# ╟─cbe4abaf-46f9-4726-97ae-cf3c378abaaf
# ╟─4f0c81f5-ce5f-4f73-a528-9feff4a7fc14
# ╠═e81de385-0070-49a9-a889-8fcf9d9e2951
# ╟─8249b820-8fb1-45d4-a95c-9c81e62e8216
# ╟─6503b377-b2d5-48be-90a4-97947afb4e5f
# ╟─53a571dd-cac7-432a-869c-b93a8fe05e17
# ╟─2e5c7f59-dd35-4846-815a-b92eabeee089
# ╟─7b326477-43b6-4a6e-8862-12e8b70e1ad9
# ╟─47cd7560-a29e-4b55-bef5-28daa1cdb834
# ╟─59ab4431-ea4d-4707-9a42-d50eafa40b56
# ╟─3630b412-beeb-455a-a4b8-1e1d50860266
# ╟─ba6347d6-4ad0-403b-824f-dcf290a7c002
# ╟─a2f4975d-eeee-4a2d-97dd-dd0cfd29d665
# ╟─32d411e9-b01d-4ad2-b4aa-2f091034e6c0
# ╟─e5b83421-dd94-43ad-84eb-ca558bff6a2d
# ╟─118642ad-1aad-4f91-8da0-55a417b67750
# ╟─45171ecc-9d34-4ab6-a00b-ec9c9afc33f8
# ╟─6cd60f7d-d7ce-4189-a2dd-e47ce6825741
# ╟─ff3fc929-f448-41be-8f60-65de33dff36a
# ╟─5ec2649e-9988-4f38-896a-64ef6ed91d82
# ╟─aa8475c3-c68b-4200-8634-ace33f525417
# ╟─aa22f905-b69d-405e-b09a-a765d60f6079
# ╟─2dcbaac0-2fad-4292-ad31-8188a60876da
# ╟─6bad4f5f-806f-480a-ae16-2582761ce5e3
# ╟─e200dde3-9033-45b5-bfe0-2d03753b2c11
# ╟─d7a4b342-ef0a-44f9-b88d-bbb04483e8b3
# ╟─908f19a2-6d32-4776-95eb-b249a8155ddc
# ╟─05fc1fae-b378-4c39-b060-74ca635745ec
# ╟─d0573bf9-0fd6-4512-bc13-17aa23a3265b
# ╟─d8998d5f-65b2-4850-aef9-f19ecc192eca
# ╟─6b4c180c-9a12-4e3d-9336-1431e7c5875a
# ╟─0eb66cc9-93c0-4f07-b31b-a9bf9000260e
# ╟─85bf9f92-30f1-4e05-8d07-d8e481f20ccb
# ╟─fbba0a91-9f48-4d91-90e7-f6a7df3227f9
# ╟─7fc81b9f-73ed-4780-9204-ddf39467e58f
# ╟─d08ae188-937f-474b-92d7-cb8eeda063fe
# ╟─7ee8cfc9-26b2-4fe4-8263-1f4d2f7c276d
# ╠═db9a97b1-f76d-4f51-96c6-0159469c5adb
# ╟─1669f5e3-efe1-4b79-a2b6-11ed7476a2a1
# ╠═5000f4c3-5416-4e53-88ae-e30d8d09827e
# ╠═2413c061-89de-403f-8011-e458f5a9859d
# ╟─d4704779-9261-478b-bbf6-551220783e12
# ╠═4be065b5-0841-4d54-b9ab-d6770d4d9d94
# ╠═e850e53d-cf61-4fc7-9cb3-e318ae957f0b
# ╟─a267ea5f-8bd5-4ee0-9c8d-47e2d3b81692
# ╟─c02520a3-3375-4d83-a0dc-1aeac2aa7d5f
# ╟─2e18fc92-4185-493b-9ce8-cca63dad7d2d
# ╟─acc7b185-e4df-4aca-aa42-554215065384
# ╟─00c9e973-7e06-4483-bf4c-be7374707118
# ╟─83ff3fc3-bcd8-4235-a42f-1d75c7d6aa5b
# ╟─b308e270-6b40-4946-ac92-c705823f2c1e
# ╟─e483b3d4-d01c-4a98-8e68-e8120a7d95a7
# ╟─74017e78-0f02-41bb-a160-5f2d26c18268
# ╟─0b3735c2-695c-4225-843e-16ca17aac0eb
# ╟─d7942b37-f821-494a-8f18-5f267aa3457a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002

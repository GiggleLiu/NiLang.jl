### A Pluto.jl notebook ###
# v0.12.21

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

# ╔═╡ 845b7d0a-7ca2-11eb-2683-cd370811bf68
using NiLang

# ╔═╡ 88912bea-7ca2-11eb-1cde-db111d594c20
using Viznet, PlutoUI, Compose

# ╔═╡ f42405a8-7ca2-11eb-133b-df4f1ce9bb19
html"""<div align="center"><a target="_blank" href="https://raw.githubusercontent.com/GiggleLiu/NiLang.jl/master/notebooks/margolus.jl">download this notebook</a></div>"""

# ╔═╡ 66e972c2-7ca2-11eb-0a7a-a9305dd20511
md"# BBMCA - NiLang implementation"

# ╔═╡ c8722cea-7ca5-11eb-3e7d-0f39322ea40a
md"Check [Physics-like models of computation](https://www.sciencedirect.com/science/article/abs/pii/0167278984902525) (Norman Margolus, 1984) for theories about Billiard ball celluar automata (BBMCA)."

# ╔═╡ 845d1458-7ca2-11eb-0e69-11a10b9894a8
@i function load_and_clear!(x, config, i, j)
	m, n ← size(config)
	x ⊻= config[i,j]  # to make it faster, should put `@inbounds` before it
	config[i,j] ⊻= x
	x ⊻= config[i,mod1(j+1, n)] << 1
	config[i,mod1(j+1, n)] ⊻= x >> 1
	x ⊻= config[mod1(i+1, m),j] << 2
	config[mod1(i+1, m),j] ⊻= x >> 2
	x ⊻= config[mod1(i+1, m),mod1(j+1, n)] << 3
	config[mod1(i+1, m),mod1(j+1, n)] ⊻= x >> 3
end

# ╔═╡ 84680136-7ca2-11eb-1df5-ffa0b4b9d126
@i function margolus_rule(y, x)
	# remove reversibility check to make it run faster
    @invcheckoff if x==6
        y ⊻= 9
    elseif x==9
        y ⊻= 6
    elseif x==4
        y ⊻= 2
    elseif x==2
        y ⊻= 4
    elseif x==1
        y ⊻= 8
    elseif x==8
        y ⊻= 1
    else
        y ⊻= x
    end
end

# ╔═╡ 845c1292-7ca2-11eb-3e56-996aa5229b4e
@i function update_bbmca!(config, iseven)
	# computing offsets, and borrow some ancillas from system
	@routine begin
		offset ← 1
		m, n ← size(config)
		if !iseven
			offset += 1
		end
	end
	for j=offset:2:n
		for i=offset:2:m
			x ← 0
			y ← 0
			# load block to `x` and clean up original data
			load_and_clear!(x, config, i, j)
			# compute new config to `y`
			margolus_rule(y, x)
			# clean up `x` with the following observation:
			# applying margolus rule twice restores the configuration
			margolus_rule(x, y)
			# store `y` to block
			(~load_and_clear!)(y, config, i, j)
			# ancillas `x` and `y` are returned to the pool automatically
		end
	end
	# uncompute `offset`
	~@routine
end

# ╔═╡ 34006d60-7ca3-11eb-3c51-c9758394b838
md"# Visualization"

# ╔═╡ 846e21da-7ca2-11eb-05a3-51e22ed04147
function showconfig(ba::AbstractMatrix)
    m, n = size(ba, 1), size(ba, 2)
    lt = Viznet.SquareLattice(n, m)
    brush1 = nodestyle(:square, fill("black"), stroke("#888888"), linewidth(unit(lt)*mm); r=unit(lt)/2.2)
    brush0 = nodestyle(:square, fill("white"), stroke("#888888"), linewidth(unit(lt)*mm); r=unit(lt)/2.2)
    canvas() do
        for i=1:m, j=1:n
            (ba[i, j] == 1 ? brush1 : brush0) >> lt[j,i]
        end
    end
end

# ╔═╡ 846e8170-7ca2-11eb-351e-eb45c629f6a6
@bind btn Clock(0.1)

# ╔═╡ 84730f04-7ca2-11eb-3b13-3fd82a9e2109
# initial configuration
config = let
	x=zeros(Int, 10, 10)
	x[1,1] = 1
	x
end;

# ╔═╡ 84763e9c-7ca2-11eb-356c-c391966cdc98
# parity - Note: BBMCA is a two state CA
bbmca_parity = Ref(true)

# ╔═╡ 847711e6-7ca2-11eb-326a-15f6bfc05347
let
	btn
	# update
	update_bbmca!(config, bbmca_parity[])
	# change parity
	bbmca_parity[] = !(bbmca_parity[])
	# visualize
	Compose.set_default_graphic_size(10cm, 10cm)
	showconfig(config)
end

# ╔═╡ Cell order:
# ╟─f42405a8-7ca2-11eb-133b-df4f1ce9bb19
# ╟─66e972c2-7ca2-11eb-0a7a-a9305dd20511
# ╟─c8722cea-7ca5-11eb-3e7d-0f39322ea40a
# ╠═845b7d0a-7ca2-11eb-2683-cd370811bf68
# ╠═845c1292-7ca2-11eb-3e56-996aa5229b4e
# ╠═845d1458-7ca2-11eb-0e69-11a10b9894a8
# ╠═84680136-7ca2-11eb-1df5-ffa0b4b9d126
# ╟─34006d60-7ca3-11eb-3c51-c9758394b838
# ╠═88912bea-7ca2-11eb-1cde-db111d594c20
# ╠═846e21da-7ca2-11eb-05a3-51e22ed04147
# ╟─846e8170-7ca2-11eb-351e-eb45c629f6a6
# ╠═84730f04-7ca2-11eb-3b13-3fd82a9e2109
# ╠═84763e9c-7ca2-11eb-356c-c391966cdc98
# ╠═847711e6-7ca2-11eb-326a-15f6bfc05347

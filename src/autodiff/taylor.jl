export BeijingRing, taylor_hessian, local_hessian, hdata
export rings_init, nrings, beijingring, collect_hessian

const rings = Vector{Float64}[]
rings_init() = empty!(rings)
nrings() = length(rings)

function collect_hessian()
    N = nrings()
    hess = [hdata((i,j)) for i=1:N, j=1:N]
end

function beijingring(x::Float64)
    nr = length(rings)+1
    r = zeros(Float64, nr*2-1)
    push!(rings, r)
    BeijingRing(x, 0.0, nr)
end

struct BeijingRing{T}
    x::T
    g::T
    index::Int
end

NiLang.AD.grad(hd::BeijingRing) = hd.g
NiLang.value(hd::BeijingRing) = hd.x

function hdata(h::Tuple{Int,Int})
    i1, i2 = h
    i1 > i2 ? rings[i1][i2] : rings[i2][end-i1+1]
end

hdata(h::Tuple{BeijingRing{T},BeijingRing{T}}) where T = hdata((h[1].index, h[2].index))

function NiLang.chfield(h::Tuple{Int, Int}, ::typeof(hdata), val)
    i1, i2 = h
    if i1 > i2
        rings[i1][i2] = val
    else
        rings[i2][end-i1+1] = val
    end
    h
end

function NiLang.chfield(h::Tuple{BeijingRing{T}, BeijingRing{T}}, ::typeof(hdata), val) where T
    NiLang.chfield((h[1].index, h[2].index), hdata, val)
    h
end

function NiLang.chfield(hd::BeijingRing, ::typeof(value), val)
    chfield(hd, Val(:x), val)
end
function NiLang.chfield(hd::BeijingRing, ::typeof(grad), val)
    chfield(hd, Val(:g), val)
end

function Base.zero(x::BeijingRing{T}) where T
    zero(BeijingRing{T})
end

function Base.zero(x::Type{BeijingRing{T}}) where T
    beijingring(zero(T))
end


function NiLangCore.deanc(x::BeijingRing, val::BeijingRing)
    pop!(NiLang.AD.rings)
    pop!(NiLang.AD.rings)
    value(x) == value(val)
end

@i function NEG(x!::BeijingRing)
    NEG(x!.x)
    # hessian from hessian
    for i=1:nrings()
        NEG(hdata((x!.index, i)))
        NEG(hdata((i, x!.index)))
    end

    # update gradients
    NEG(grad(x!))
end

@i function CONJ(x!::BeijingRing)
    CONJ(x!.x)
    # hessian from hessian
    for i=1:nrings()
        CONJ(hdata((x!.index, i)))
        CONJ(hdata((i, x!.index)))
    end

    # update gradients
    CONJ(grad(x!))
end

@i function ⊖(identity)(out!::BeijingRing, x::BeijingRing)
    ⊖(identity)(out!.x, x.x)
    # hessian from hessian
    for i=1:nrings()
        hdata((x.index, i)) ⊕ hdata((out!.index, i))
    end
    for i=1:nrings()
        hdata((i, x.index)) ⊕ hdata((i, out!.index))
    end

    # update gradients
    grad(x) ⊕ grad(out!)
end

@i function ⊖(exp)(out!::BeijingRing{T}, x::BeijingRing) where T
    @anc expx = zero(T)
    expx += exp(x.x)
    out!.x ⊕ expx
    # hessian from hessian
    for i=1:nrings()
        hdata((x.index, i)) += expx * hdata((out!.index, i))
    end
    for i=1:nrings()
        hdata((i, x.index)) += expx * hdata((i, out!.index))
    end
    hdata((x, x)) += expx * grad(out!)

    # update gradients
    grad(x) += expx * grad(out!)
    expx -= exp(x.x)
end

@i function ⊖(log)(out!::BeijingRing{T}, x::BeijingRing) where T
    @anc g = zero(T)
    @anc h = zero(T)
    out!.x += log(x.x)

    @routine r1 begin
        g += 1.0/x.x
        h += g/x.x
        NEG(h)
    end
    # hessian from hessian
    for i=1:nrings()
        hdata((x.index, i)) += g * hdata((out!.index, i))
    end
    for i=1:nrings()
        hdata((i, x.index)) += g * hdata((i, out!.index))
    end
    hdata((x, x)) += h * grad(out!)

    # update gradients
    grad(x) += g * grad(out!)

    ~@routine r1
end

@i function ⊖(sin)(out!::BeijingRing{T}, x::BeijingRing) where T
    @anc sinx = zero(T)
    @anc cosx = zero(T)
    sinx += sin(x.x)
    cosx += cos(x.x)
    out!.x ⊕ sinx
    # hessian from hessian
    for i=1:nrings()
        hdata((x.index, i)) += cosx * hdata((out!.index, i))
    end
    for i=1:nrings()
        hdata((i, x.index)) += cosx * hdata((i, out!.index))
    end
    hdata((x, x)) -= sinx * grad(out!)

    # update gradients
    grad(x) += cosx * grad(out!)
    sinx -= sin(x.x)
    cosx -= cos(x.x)
end

@i function ⊖(cos)(out!::BeijingRing{T}, x::BeijingRing) where T
    @anc sinx = zero(T)
    @anc cosx = zero(T)
    sinx += sin(x.x)
    cosx += cos(x.x)
    out!.x ⊕ cosx
    # hessian from hessian
    for i=1:nrings()
        hdata((x.index, i)) -= sinx * hdata((out!.index, i))
    end
    for i=1:nrings()
        hdata((i, x.index)) -= sinx * hdata((i, out!.index))
    end
    hdata((x, x)) -= cosx * grad(out!)

    # update gradients
    grad(x) -= sinx * grad(out!)
    sinx -= sin(x.x)
    cosx -= cos(x.x)
end

@i function SWAP(x!::BeijingRing, y!::BeijingRing)
    SWAP(x!.x, y!.x)
    # hessian from hessian
    for i=1:nrings()
        SWAP(hdata((x!.index, i)), hdata((y!.index, i)))
    end
    for i=1:nrings()
        SWAP(hdata((i, x!.index)), hdata((i, y!.index)))
    end

    # update gradients
    SWAP(grad(x!), grad(y!))
end

# dL^2/dx/dy = ∑(dL^2/da/db)*da/dx*db/dy
# https://arxiv.org/abs/1206.6464
@i function ⊖(*)(out!::BeijingRing, x, y)
    ⊖(*)(out!.x, value(x), value(y))
    # hessian from hessian
    for i=1:nrings()
        if (x isa BeijingRing, ~)
            hdata((x.index, i)) += value(y) * hdata((out!.index, i))
        end
        if (y isa BeijingRing, ~)
            hdata((y.index, i)) += value(x) * hdata((out!.index, i))
        end
    end
    for i=1:nrings()
        if (x isa BeijingRing, ~)
            hdata((i, x.index)) += value(y) * hdata((i, out!.index))
        end
        if (y isa BeijingRing, ~)
            hdata((i, y.index)) += value(x) * hdata((i, out!.index))
        end
    end

    # hessian from jacobian
    if (x isa BeijingRing && y isa BeijingRing, ~)
        hdata((x, y)) ⊕ grad(out!)
        hdata((y, x)) ⊕ grad(out!)
    end

    # update gradients
    if (x isa BeijingRing, ~)
        grad(x) += grad(out!) * value(y)
    end
    if (y isa BeijingRing, ~)
        grad(y) += value(x) * grad(out!)
    end
end

@i function ⊖(/)(out!::BeijingRing{T}, x, y) where T
    ⊖(/)(out!.x, value(x), value(y))
    @anc binv = zero(T)
    @anc binv2 = zero(T)
    @anc binv3 = zero(T)
    @anc a3 = zero(T)
    @anc xjac = zero(T)
    @anc yjac = zero(T)
    @anc yyjac = zero(T)
    @anc xyjac = zero(T)

    @routine jacs begin
        # compute dout/dx and dout/dy
        xjac += 1.0/value(y)
        binv2 += xjac^2
        binv3 += xjac^3
        yjac -= value(x)*binv2
        a3 += value(x)*binv3
        yyjac += 2*a3
        xyjac ⊖ binv2
    end
    # hessian from hessian
    for i=1:nrings()
        if (x isa BeijingRing, ~)
            hdata((x.index, i)) += xjac * hdata((out!.index, i))
        end
        if (y isa BeijingRing, ~)
            hdata((y.index, i)) += yjac * hdata((out!.index, i))
        end
    end
    for i=1:nrings()
        if (x isa BeijingRing, ~)
            hdata((i, x.index)) += xjac * hdata((i, out!.index))
        end
        if (y isa BeijingRing, ~)
            hdata((i, y.index)) += yjac * hdata((i, out!.index))
        end
    end

    # hessian from jacobian
    if (x isa BeijingRing && y isa BeijingRing, ~)
        hdata((y.index, y.index)) += yyjac*grad(out!)
        hdata((x.index, y.index)) += xyjac*grad(out!)
        hdata((y.index, x.index)) += xyjac*grad(out!)
    end

    # update gradients
    if (x isa BeijingRing, ~)
        grad(x) += grad(out!) * xjac
    end
    if (y isa BeijingRing, ~)
        grad(y) += yjac * grad(out!)
    end

    ~@routine jacs
end

@i function ⊖(^)(out!::BeijingRing{T}, x, n) where T
    ⊖(^)(out!.x, value(x), value(n))
    @anc logx = zero(T)
    @anc logx2 = zero(T)
    @anc powerxn = zero(T)
    @anc anc1 = zero(T)
    @anc anc2 = zero(T)
    @anc xjac = zero(T)
    @anc njac = zero(T)
    @anc hxn = zero(T)
    @anc hxx = zero(T)
    @anc hnn = zero(T)
    @anc nminus1 = zero(T)

    # compute jacobians
    @routine getjac begin
        nminus1 ⊕ value(n)
        nminus1 ⊖ 1
        powerxn += value(x)^value(n)
        logx += log(value(x))
        out!.x ⊕ powerxn

        # dout!/dx = n*x^(n-1)
        anc1 += value(x)^nminus1
        xjac += anc1 * value(n)
        # dout!/dn = logx*x^n
        njac += logx*powerxn

        # for hessian
        logx2 += logx^2
        anc2 += xjac/value(x)
        hxn ⊕ anc1
        hxn += xjac * logx
        hxx += anc2 * nminus1
        hnn += logx2 * powerxn
    end

    # hessian from hessian
    for i=1:nrings()
        if (x isa BeijingRing, ~)
            hdata((i, x.index)) += hdata((i, out!.index)) * xjac
        end
        if (n isa BeijingRing, ~)
            hdata((i, n.index)) += hdata((i, out!.index)) * njac
        end
    end
    for i=1:nrings()
        if (x isa BeijingRing, ~)
            hdata((x.index, i)) += hdata((out!.index, i)) * xjac
        end
        if (n isa BeijingRing, ~)
            hdata((n.index, i)) += hdata((out!.index, i)) * njac
        end
    end

    # hessian from jacobian
    # Dnn = x^n*log(x)^2
    # Dxx = (-1 + n)*n*x^(-2 + n)
    # Dxn = Dnx = x^(-1 + n) + n*x^(-1 + n)*log(x)
    if (x isa BeijingRing && n isa BeijingRing, ~)
        hdata((x, x)) += hxx * grad(out!)
        hdata((n, n)) += hnn * grad(out!)
        hdata((x, n)) += hxn * grad(out!)
        hdata((n, x)) += hxn * grad(out!)
    end

    # update gradients
    if (x isa BeijingRing, ~)
        grad(x) += grad(out!) * xjac
    end
    if (n isa BeijingRing, ~)
        grad(n) += grad(out!) * njac
    end

    ~@routine getjac
end

@i function IROT(a!::BeijingRing{T}, b!, θ) where T
    @anc s = zero(T)
    @anc c = zero(T)
    @anc ca = zero(T)
    @anc sb = zero(T)
    @anc sa = zero(T)
    @anc cb = zero(T)
    @anc θ2 = zero(T)
    IROT(value(a!), value(b!), value(θ))

    @routine temp begin
        θ2 ⊖ value(θ)
        θ2 ⊖ π/2
        s += sin(value(θ))
        c += cos(value(θ))
        ca += c * value(a!)
        sb += s * value(b!)
        sa += s * value(a!)
        cb += c * value(b!)
    end

    # update gradient, #1
    for i=1:nrings()
        ROT(hdata((i, a!.index)), hdata((i, b!.index)), θ2)
        hdata((i, θ.index)) += value(a!) * hdata((i, a!.index))
        hdata((i, θ.index)) += value(b!) * hdata((i, b!.index))
        ROT(hdata((i, a!.index)), hdata((i, b!.index)), π/2)
    end
    for i=1:nrings()
        ROT(hdata((a!.index, i)), hdata((b!.index, i)), θ2)
        hdata((θ.index, i)) += value(a!) * hdata((a!.index, i))
        hdata((θ.index, i)) += value(b!) * hdata((b!.index, i))
        ROT(hdata((a!.index, i)), hdata((b!.index, i)), π/2)
    end

    # update local hessian
    hdata((a!, θ)) -= s * grad(a!)
    hdata((b!, θ)) -= c * grad(a!)
    hdata((θ, a!)) -= s * grad(a!)
    hdata((θ, b!)) -= c * grad(a!)
    hdata((θ, θ)) -= ca * grad(a!)
    hdata((θ, θ)) += sb * grad(a!)

    hdata((a!, θ)) += c * grad(b!)
    hdata((b!, θ)) -= s * grad(b!)
    hdata((θ, a!)) += c * grad(b!)
    hdata((θ, b!)) -= s * grad(b!)
    hdata((θ, θ)) -= sa * grad(b!)
    hdata((θ, θ)) -= cb * grad(b!)

    # update gradients
    ROT(grad(a!), grad(b!), θ2)
    grad(θ) += value(a!) * grad(a!)
    grad(θ) += value(b!) * grad(b!)
    ROT(grad(a!), grad(b!), π/2)

    ~@routine temp
end

function local_hessian(f, args; kwargs=())
    nargs = length(args)
    hes = zeros(nargs,nargs,nargs)
    @instr f(args...)
    for j=1:nargs
        rings_init()
        largs = [beijingring(arg) for arg in args]
        @instr grad(largs[j]) ⊕ 1.0
        @instr (~f)(largs...)
        hes[:,:,j] .= collect_hessian()
    end
    hes
end

function taylor_hessian(f, args::Tuple; kwargs=Dict())
    @assert count(x -> x isa Loss, args) == 1
    N = length(args)

    iloss = 0
    for i=1:length(args)
        if tget(args,i) isa Loss
            iloss += identity(i)
        end
    end
    @instr (~Loss)(tget(args, iloss))

    @instr f(args...)
    rings_init()
    args = [beijingring(x) for x in args]
    @instr grad(args[iloss]) ⊕ 1.0
    @instr (~f)(args...)
    @show args
    collect_hessian()
end

macro nohess(ex)
    @match ex begin
        :($f($(args...))) => begin
            newargs = []
            for arg in args
                push!(newargs, @match arg begin
                    :($x::BeijingRing) => :($x.x)
                    :($x::BeijingRing{$tp}) => :($x.x)
                    _ => arg
                end
                )
            end
            esc(quote
                @i function $f($(args...))
                    $f($(newargs...))
                end
            end)
        end
        _ => error("expect `f(args...)`, got $ex")
    end
end

@nohess ⊖(identity)(out!::BeijingRing, x)
@nohess ⊖(identity)(out!, x::BeijingRing)
for op in [:*, :/, :^]
    @eval @nohess ⊖($op)(out!::BeijingRing, x::Number, y::Number)
    @eval @nohess ⊖($op)(out!::Number, x, y::BeijingRing)
    @eval @nohess ⊖($op)(out!::Number, x::BeijingRing, y::BeijingRing)
    @eval @nohess ⊖($op)(out!::Number, x::BeijingRing, y)
end

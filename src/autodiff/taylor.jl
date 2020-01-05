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

# dL^2/dx/dy = ∑(dL^2/da/db)*da/dx*db/dy
# https://arxiv.org/abs/1206.6464
@i function ⊖(*)(out!::BeijingRing, x::BeijingRing, y::BeijingRing)
    ⊖(*)(out!.x, x.x, y.x)
    # hessian from hessian
    for i=1:nrings()
        hdata((x.index, i)) += y.x * hdata((out!.index, i))
        hdata((y.index, i)) += x.x * hdata((out!.index, i))
    end
    for i=1:nrings()
        hdata((i, x.index)) += y.x * hdata((i, out!.index))
        hdata((i, y.index)) += x.x * hdata((i, out!.index))
    end

    # hessian from jacobian
    hdata((x, y)) ⊕ grad(out!)
    hdata((y, x)) ⊕ grad(out!)

    # update gradients
    grad(x) += grad(out!) * value(y)
    grad(y) += value(x) * grad(out!)
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

@i function ⊖(/)(out!::BeijingRing{T}, x::BeijingRing{T}, y::BeijingRing{T}) where T
    ⊖(/)(out!.x, x.x, y.x)
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
        hdata((x.index, i)) += xjac * hdata((out!.index, i))
        hdata((y.index, i)) += yjac * hdata((out!.index, i))
    end
    for i=1:nrings()
        hdata((i, x.index)) += xjac * hdata((i, out!.index))
        hdata((i, y.index)) += yjac * hdata((i, out!.index))
    end

    # hessian from jacobian
    hdata((y.index, y.index)) += yyjac*grad(out!)
    hdata((x.index, y.index)) += xyjac*grad(out!)
    hdata((y.index, x.index)) += xyjac*grad(out!)

    # update gradients
    grad(x) += grad(out!) * xjac
    grad(y) += yjac * grad(out!)

    ~@routine jacs
end

@i function ⊖(^)(out!::BeijingRing{T}, x::BeijingRing{T}, n::BeijingRing{T}) where T
    ⊖(^)(out!.x, x.x, n.x)
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
        nminus1 ⊕ n.x
        nminus1 ⊖ 1
        powerxn += x.x^n.x
        logx += log(x.x)
        out!.x ⊕ powerxn

        # dout!/dx = n*x^(n-1)
        anc1 += x^nminus1
        xjac += anc1 * value(n)
        # dout!/dn = logx*x^n
        njac += logx*powerxn

        # for hessian
        logx2 += logx^2
        anc2 += xjac/x
        hxn ⊕ anc1
        hxn += xjac * logx
        hxx += anc2 * nminus1
        hnn += logx2 * powerxn
    end

    # hessian from hessian
    for i=1:nrings()
        hdata((i, x.index)) += hdata((i, out!.index)) * xjac
        hdata((i, n.index)) += hdata((i, out!.index)) * njac
    end
    for i=1:nrings()
        hdata((x.index, i)) += hdata((out!.index, i)) * xjac
        hdata((n.index, i)) += hdata((out!.index, i)) * njac
    end

    # hessian from jacobian
    # Dnn = x^n*log(x)^2
    # Dxx = (-1 + n)*n*x^(-2 + n)
    # Dxn = Dnx = x^(-1 + n) + n*x^(-1 + n)*log(x)
    hdata((x, x)) += hxx * grad(out!)
    hdata((n, n)) += hnn * grad(out!)
    hdata((x, n)) += hxn * grad(out!)
    hdata((n, x)) += hxn * grad(out!)

    # update gradients
    grad(x) += grad(out!) * xjac
    grad(n) += grad(out!) * njac

    ~@routine getjac
end

@i function IROT(a!::BeijingRing{T}, b!::BeijingRing{T}, θ::BeijingRing{T}) where T
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

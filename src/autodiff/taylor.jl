export HessianData, taylor_hessian

struct HessianData{T}
    x::T
    gradient::AbstractVector{T}
    hessian::AbstractArray{T}
    index::Int
end

size_paramspace(hd::HessianData) = length(hd.gradient)
NiLang.AD.grad(hd::HessianData) = hd.gradient[hd.index]
NiLang.value(hd::HessianData) = hd.x
hrow(hd::HessianData) = view(hd.hessian, hd.index, :)
hcol(hd::HessianData) = view(hd.hessian, :, hd.index)
function NiLang.chfield(hd::HessianData, ::typeof(hrow), val)
    hrow(hd) .= val
    hd
end
function NiLang.chfield(hd::HessianData, ::typeof(hcol), val)
    hcol(hd) .= val
    hd
end
function NiLang.chfield(hd::HessianData, ::typeof(value), val)
    chfield(hd, Val(:x), val)
end
function NiLang.chfield(hd::HessianData, ::typeof(grad), val)
    hd.gradient[hd.index] = val
    hd
end

# dL^2/dx/dy = ∑(dL^2/da/db)*da/dx*db/dy
# https://arxiv.org/abs/1206.6464
@i function ⊖(*)(out!::HessianData, x::HessianData, y::HessianData)
    ⊖(*)(out!.x, x.x, y.x)
    # hessian from hessian
    for i=1:size_paramspace(out!)
        hrow(x)[i] += y.x * hrow(out!)[i]
        hrow(y)[i] += x.x * hrow(out!)[i]
        hcol(x)[i] += y.x * hcol(out!)[i]
        hcol(y)[i] += x.x * hcol(out!)[i]
    end

    # hessian from jacobian
    out!.hessian[x.index, y.index] ⊕ grad(out!)
    out!.hessian[y.index, x.index] ⊕ grad(out!)

    # update gradients
    grad(x) += grad(out!) * value(y)
    grad(y) += value(x) * grad(out!)
end

@i function ⊖(identity)(out!::HessianData, x::HessianData)
    ⊖(identity)(out!.x, x.x)
    # hessian from hessian
    for i=1:size_paramspace(out!)
        hrow(x)[i] ⊕ hrow(out!)[i]
        hcol(x)[i] ⊕ hcol(out!)[i]
    end

    # update gradients
    grad(x) ⊕ grad(out!)
end

@i function ⊖(/)(out!::HessianData{T}, x::HessianData{T}, y::HessianData{T}) where T
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
    for i=1:size_paramspace(out!)
        hrow(x)[i] += xjac * hrow(out!)[i]
        hrow(y)[i] += yjac * hrow(out!)[i]
        hcol(x)[i] += xjac * hcol(out!)[i]
        hcol(y)[i] += yjac * hcol(out!)[i]
    end

    # hessian from jacobian
    out!.hessian[y.index, y.index] += yyjac*grad(out!)
    out!.hessian[x.index, y.index] += xyjac*grad(out!)
    out!.hessian[y.index, x.index] += xyjac*grad(out!)

    @safe @show grad(out!), xjac, yjac, grad(x), grad(y)
    # update gradients
    grad(x) += grad(out!) * xjac
    grad(y) += yjac * grad(out!)

    ~@routine jacs
end

@i function ⊖(^)(out!::HessianData{T}, x::HessianData{T}, n::HessianData{T}) where T
    ⊖(^)(out!.x, x.x, n.x)
    @anc logx = zero(T)
    @anc logx2 = zero(T)
    @anc powerxn = zero(T)
    @anc anc1 = zero(T)
    @anc anc2 = zero(T)
    @anc xjac = zero(T)
    @anc njac = zero(T)
    @anc hxn = zero(T)
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
    end

    # hessian from hessian
    for i=1:size_paramspace(out!)
        hcol(x)[i] += hcol(out!)[i] * xjac
        hrow(x)[i] += hrow(out!)[i] * xjac
        hrow(n)[i] += hrow(out!)[i] * njac
        hcol(n)[i] += hcol(out!)[i] * njac
    end

    # hessian from jacobian
    # Dnn = x^n*log(x)^2
    # Dxx = (-1 + n)*n*x^(-2 + n)
    # Dxn = Dnx = x^(-1 + n) + n*x^(-1 + n)*log(x)
    out!.hessian[x.index, x.index] += anc2 * nminus1
    out!.hessian[n.index, n.index] += logx2 * powerxn
    out!.hessian[x.index, n.index] ⊕ hxn
    out!.hessian[n.index, x.index] ⊕ hxn

    # update gradients
    grad(x) += grad(out!) * xjac
    grad(n) += grad(out!) * njac

    ~@routine getjac
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
    grad = zeros(N); grad[iloss] = 1.0
    hess = zeros(N, N)
    args = [HessianData(x, grad, hess, i) for (i,x) in enumerate(args)]
    @instr (~f)(args...)
    args[1].hessian
end

@i function IROT(a!::HessianData{T}, b!::HessianData{T}, θ::HessianData{T}) where T
    @anc s = zero(T)
    @anc c = zero(T)
    IROT(value(a!), value(b!), value(θ))

    NEG(value(θ))
    value(θ) ⊖ π/2

    # update gradient and hessian, #1
    ROT(grad(a!), grad(b!), value(θ))
    grad(θ) += value(a!) * grad(a!)
    grad(θ) += value(b!) * grad(b!)
    for i=1:size_paramspace(a!)
        ROT(hcol(a!)[i], hcol(b!)[i], value(θ))
        ROT(hrow(a!)[i], hrow(b!)[i], value(θ))
        hcol(θ)[i] += value(a!) * hcol(a!)[i]
        hrow(θ)[i] += value(a!) * hrow(a!)[i]
        hcol(θ)[i] += value(b!) * hcol(b!)[i]
        hrow(θ)[i] += value(b!) * hrow(b!)[i]
    end

    value(θ) ⊕ π/2
    NEG(value(θ))

    # update gradient and hessian, #2
    ROT(grad(a!), grad(b!), π/2)
    for i=1:size_paramspace(a!)
        ROT(hcol(a!)[i], hcol(b!)[i], π/2)
        ROT(hrow(a!)[i], hrow(b!)[i], π/2)
    end

    # update local hessian
    s += sin(value(θ))
    c += cos(value(θ))
    a!.hessian[a!.index, θ.index] ⊖ s
    a!.hessian[b!.index, θ.index] ⊖ c
    a!.hessian[θ.index, a!.index] ⊖ s
    a!.hessian[θ.index, b!.index] ⊖ c
    a!.hessian[θ.index, θ.index] -= c * value(a!)
    a!.hessian[θ.index, θ.index] += s * value(b!)
    s -= sin(value(θ))
    c -= cos(value(θ))
end

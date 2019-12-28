using NiLang, NiLang.AD

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

const _hmat_mul = [0.0 0 0; 0 0 1; 0 1 0]
Hmat(::typeof(⊕(*)), ) = _hmat_mul
const _hmat_identity = zeros(2,2)
Hmat(::typeof(⊕(identity))) = _hmat_identity

# dL^2/dx/dy = ∑(dL^2/da/db)*da/dx*db/dy
# https://arxiv.org/abs/1206.6464
@i function ⊖(*)(out!::HessianData, x::HessianData, y::HessianData)
    ⊖(*)(out!.x, x.x, y.x)
    # hessian from hessian
    for i=1:size_paramspace(out!)
        hrow(x)[i] += y.x * hrow(out!)[i]
        hrow(y)[i] += x.x * hrow(out!)[i]
    end
    for i=1:size_paramspace(out!)
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
    end
    for i=1:size_paramspace(out!)
        hcol(x)[i] ⊕ hcol(out!)[i]
    end

    # update gradients
    grad(x) ⊕ grad(out!)
end

@i function ⊖(/)(out!::HessianData, x::HessianData, y::HessianData)
    ⊖(/)(out!.x, x.x, y.x)
    # hessian from hessian
    for i=1:size_paramspace(out!)
        hrow(x)[i] += y.x * hrow(out!)[i]
        hrow(y)[i] += x.x * hrow(out!)[i]
    end
    for i=1:size_paramspace(out!)
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

@i function ⊖(^)(out!::HessianData, x::HessianData, y::HessianData)
    ⊖(^)(out!.x, x.x, y.x)
    # hessian from hessian
    for i=1:size_paramspace(out!)
        hrow(x)[i] += y.x * hrow(out!)[i]
        hrow(y)[i] += x.x * hrow(out!)[i]
    end
    for i=1:size_paramspace(out!)
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

@testset "hessian *" begin
    hdata = zeros(3,3)
    gdata = [1.0, 0, 0]
    out! = HessianData(6.0, gdata, hdata, 1)
    a = HessianData(2.0, gdata, hdata, 2)
    b = HessianData(3.0, gdata, hdata, 3)
    @instr ⊖(*)(out!, a, b)
    @test out!.x == 0
    @test out!.gradient == [1, 3.0, 2.0]
    @test out!.hessian == [0 0 0; 0 0 1; 0 1 0.0]
end

@testset "hessian +" begin
    hdata = zeros(2,2)
    gdata = [1.0, 0]
    out! = HessianData(3.0, gdata, hdata, 1)
    a = HessianData(2.0, gdata, hdata, 2)
    @instr ⊖(identity)(out!, a)
    @test out!.x == 1
    @test out!.gradient == [1, 1.0]
    @test out!.hessian == [0 0; 0 0]
end

using Test
@testset "HessianData" begin
    hdata = zeros(3,3)
    gdata = [1.0, 0, 0]
    out! = HessianData(6.0, gdata, hdata, 1)
    a = HessianData(2.0, gdata, hdata, 2)
    b = HessianData(3.0, gdata, hdata, 3)

    @test size_paramspace(out!) == 3

    @test chfield(out!, Val(:x), 0.5) == HessianData(0.5, gdata, hdata, 1)
    @test chfield(out!, value, 0.6) == HessianData(0.6, gdata, hdata, 1)
    chfield(a, grad, 0.5)
    @test gdata == [1.0,0.5,0.0]
    chfield(b, hrow, [1,2,3.0])
    @test hdata == [0 0 0; 0 0 0; 1.0 2.0 3.0]
    chfield(b, hcol, [1,2,3.0])
    @test hdata == [0 0 1; 0 0 2; 1 2 3.0]
end

nhessian(⊕(*), (Loss(0.0), 1.0, 2.0))

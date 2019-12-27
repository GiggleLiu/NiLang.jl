using NiLang, NiLang.AD

function hessian(f, args; kwargs=(), η=1e-5)
    largs = Any[args...]
    narg = length(largs)
    res = zeros(narg, narg)
    for i = 1:narg
        @instr value(largs[i]) ⊕ η/2
        gpos = gradient(f, (largs...,); kwargs=kwargs)
        @instr value(largs[i]) ⊖ η
        gneg = gradient(f, (largs...,); kwargs=kwargs)
        @instr value(largs[i]) ⊕ η/2
        res[:,i] .= (gpos .- gneg)./η
    end
    return res
end

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
end

# dL^2/dx/dy = ∑(dL^2/da/db)*da/dx*db/dy
@i function ⊖(*)(out!::HessianData, a::HessianData, b::HessianData)
    ⊖(*)(out!.x, a.x, b.x)
    @safe println("a")
    for i=1:size_paramspace(out!)
        @safe @show a, b, out!
        hrow(a)[i] += b.x * hrow(out!)[i]
        @safe @show a, b, out!
        hrow(b)[i] += a.x * hrow(out!)[i]
        @safe @show a, b, out!
    end
    for i=1:size_paramspace(out!)
        hcol(a)[i] += b.x * hcol(out!)[i]
        hcol(b)[i] += a.x * hcol(out!)[i]
    end
end

hdata = zeros(3,3)
gdata = [1.0, 0, 0]
out! = HessianData(6.0, gdata, hdata, 1)
a = HessianData(2.0, gdata, hdata, 2)
b = HessianData(3.0, gdata, hdata, 3)
⊖(*)(out!, a, b)

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

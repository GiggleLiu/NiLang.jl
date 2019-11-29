using NiLangCore

const GLOBAL_PHASE = Ref(0.0)

export qphase
qphase() = GLOBAL_PHASE[]
qphase(θ) = GLOBAL_PHASE[] = θ

"""
Rx = cos(θ/2) + im*sin(θ/2)*X
2×2 Array{Basic,2}:
    cos((1/2)*θ)  -I*sin((1/2)*θ)
 -I*sin((1/2)*θ)     cos((1/2)*θ)
"""
function Rx!(bit::Bool, θ)
    s = sin(θ/2)
    if rand() < s^2
        qphase(qphase()+(s > 0 ? π/2 : -π/2))
        !bit, θ
    else
        bit, θ
    end
end
iRx!(bit::Bool, θ) = Rx!(bit, -θ)
@dual Rx! iRx!

"""
Ry = cos(θ/2) + im*sin(θ/2)*Y
2×2 Array{Basic,2}:
 cos((1/2)*θ)  -sin((1/2)*θ)
 sin((1/2)*θ)   cos((1/2)*θ)
"""
function Ry!(bit::Bool, θ)
    s = sin(θ/2)
    if rand() < s^2
        qphase(qphase()+(bit ⊻ (s < 0) ? π : 0.0))
        !bit, θ
    else
        bit, θ
    end
end
iRy!(bit::Bool, θ) = Ry!(bit, -θ)
@dual Ry! iRy!

"""
Rz = cos(θ/2) + im*sin(θ/2)*Z
2×2 Array{Basic,2}:
 -I*sin((1/2)*θ) + cos((1/2)*θ)                              0
                              0  I*sin((1/2)*θ) + cos((1/2)*θ)
"""
function Rz!(bit::Bool, θ)
    s = sin(θ/2)
    if bit
        qphase(qphase()+θ/2)
    else
        qphase(qphase()-θ/2)
    end
    bit, θ
end
iRz!(bit::Bool, θ) = Rz!(bit, -θ)
@dual Rz! iRz!

using Test, Random
@testset "Rx, Ry, Rz" begin
    Random.seed!(2)
    for i=1:10
        b = true
        @instr Rx!(b, π)
        @test b == false
        @instr Rx!(b, 2π)
        @test b == false
    end
    qphase(0.0)
    b = true
    @instr Rx!(b, π)
    @test qphase() == π/2

    for i=1:10
        b = true
        @instr Ry!(b, π)
        @test b == false
        @instr Ry!(b, 2π)
        @test b == false
    end
    qphase(0.0)
    b = true
    @instr Ry!(b, π)
    @test qphase() ≈ π

    for i=1:10
        b = true
        @instr Rz!(b, π)
        @test b == true
        @instr Rz!(b, 2π)
        @test b == true
    end
    qphase(0.0)
    b = true
    @instr Rz!(b, π/4)
    @test qphase() ≈ π/8

    qphase(0.0)
    b = false
    @instr Rz!(b, π/4)
    @test qphase() ≈ -π/8
end

@testset "quantum reversibility" begin
    Random.seed!(2)
    cum = [0, 0.0im]
    θ = π/4
    for i=1:100
        # compute <0|Rx(-θ)Rx(θ)|0> and <1|Rx(-θ)Rx(θ)|0>
        b = false
        qphase(0.0)
        @instr Rx!(b, θ)
        @instr (~Rx!)(b, θ)
        cum[Int(b)+1] += exp(im*qphase())
    end
    normalize!(cum)
    @test isapprox(cum[1], 1; atol=1e-1)
    @test isapprox(cum[2], 0; atol=1e-1)
end

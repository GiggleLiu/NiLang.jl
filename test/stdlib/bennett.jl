using Test
using NiLang, NiLang.AD

@testset "integrate" begin
    FT = Float64
    n = 100
    h = FT(π/n)
    dt = FT(0.01)
    α = FT(4e-2)
    @i function step!(dest::AbstractArray{T}, src::AbstractArray{T}; α, h, dt) where T
        n ← length(dest)
        @invcheckoff for i=1:n
            @routine begin
                @zeros T cum g h2 αcum
                cum += src[mod1(i+1, n)] + src[mod1(i-1, n)]
                cum -= 2*src[i]
                αcum += cum * α
                h2 += h^2
                g += αcum/h2
            end
            dest[i] += src[i]
            dest[i] += dt*g
            ~@routine
        end
    end
    x = zeros(FT, n)
    x[n÷2] = 1
    #state = Dict{Int,Vector{FT}}()
    k = 4
    nsteps = 100
    x_last = NiLang.direct_emulate(step!, FT.(x); nsteps=nsteps, α=α, h=h, dt=dt)
    _, x_last_b, _ = bennett(step!, zero(FT.(x)), FT.(x); k=k, nsteps=nsteps, α=α, h=h, dt=dt)
    @test sum(x_last_b) ≈ 1
    @test x_last ≈ x_last_b

    @i function loss(out, step, y, x; kwargs...)
        bennett((@skip! step), y, x; kwargs...)
        out += y[n÷2]
    end
    _, _, _, gx = NiLang.AD.gradient(loss, (0.0, step!, zero(x), copy(x)); iloss=1, k=k, nsteps=nsteps, α=α, h=h, dt=dt)
    x_last_2 = NiLang.direct_emulate(step!, (x2=copy(x); x2[n÷2]+=1e-5; FT.(x2)); nsteps=nsteps, α=α, h=h, dt=dt)
    @test gx[n÷2] ≈ (x_last_2 - x_last)[n÷2]/1e-5
end
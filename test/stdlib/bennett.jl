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
    N = 100
    x_last = NiLang.direct_emulate(step!, FT.(x); N=N, α=α, h=h, dt=dt)
    log1 = NiLang.BennettLog()
    log2 = NiLang.BennettLog()
    _, x_last_b, _ = bennett(step!, zero(FT.(x)), FT.(x); k=k, N=N, α=α, h=h, dt=dt, logger=log1)
    _, x_last_b2 = bennett!(step!, Dict(1=>FT.(x)); k=k, N=N, α=α, h=h, dt=dt, logger=log2)
    @test sum(x_last_b) ≈ 1
    @test x_last ≈ x_last_b
    @test x_last ≈ x_last_b2[N+1]
    @test length(log1.fcalls) > length(log2.fcalls)
    @test length(log1.fcalls) < 2*length(log2.fcalls)

    @i function loss(out, step, y, x; kwargs...)
        bennett((@skip! step), y, x; kwargs...)
        out += y[n÷2]
    end
    @i function loss2(out, step, d; N, kwargs...)
        bennett!((@skip! step), d; N, kwargs...)
        out += d[N+1][n÷2]
    end
    _, _, _, gx = NiLang.AD.gradient(loss, (0.0, step!, zero(x), copy(x)); iloss=1, k=k, N=N, α=α, h=h, dt=dt)
    _, _, gx2 = NiLang.AD.gradient(loss2, (0.0, step!, Dict(1=>copy(x))); iloss=1, k=k, N=N, α=α, h=h, dt=dt)
    x_last_2 = NiLang.direct_emulate(step!, (x2=copy(x); x2[n÷2]+=1e-5; FT.(x2)); N=N, α=α, h=h, dt=dt)
    @test gx[n÷2] ≈ (x_last_2 - x_last)[n÷2]/1e-5
    @test gx2[1][n÷2] ≈ (x_last_2 - x_last)[n÷2]/1e-5
end
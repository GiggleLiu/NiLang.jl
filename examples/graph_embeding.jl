include("Adam.jl")
using NiLang, NiLang.AD

@i function mean(out!, x)
    anc ← zero(out!)
    for i=1:length(x)
        anc += identity(x[i])
    end
    out! += anc / length(x)
    anc -= out! * length(x)
end

@i function var_and_mean(var!, mean!, v::AbstractVector{T}) where T
    cum ← zero(var!)
    n ← length(v)-1
    mean(mean!, v)
    for i=1:length(v)
        v[i] -= identity(mean!)
        cum += v[i] ^ 2
        v[i] += identity(mean!)
    end
    var! += cum / n
    cum -= var! * n
end

@i function sqdistance(dist!, x1::AbstractVector{T}, x2::AbstractVector) where T
    @inbounds for i=1:length(x1)
        x1[i] -= identity(x2[i])
        dist! += x1[i] ^ 2
        x1[i] += identity(x2[i])
    end
end

@i function iloss(out!::T, x) where T
    v1 ← zero(T)
    m1 ← zero(T)
    v2 ← zero(T)
    m2 ← zero(T)
    diff ← zero(T)
    L1 ← [(1, 6), (2, 7), (3, 8), (4, 9), (5, 10),
    (1, 2), (2, 3), (3, 4), (4, 5), (1, 5), (6, 8),
    (8, 10), (7, 10), (7, 9), (6, 9)]
    L2 ← [(1, 3), (1, 4), (1, 7), (1, 8), (1, 9),
    (1, 10), (2, 4), (2, 5), (2, 6), (2, 8), (2, 9),
    (2, 10), (3, 5), (3, 6), (3, 7), (3, 9), (3, 10),
    (4, 6), (4, 7), (4, 8), (4, 10), (5, 6), (5, 7),
    (5, 8), (5, 9), (6, 7), (6, 10), (7, 8), (8, 9),
    (9, 10)]
    d1 ← zeros(T, length(L1))
    d2 ← zeros(T, length(L2))
    @routine begin
        for i=1:length(L1)
            sqdistance(d1[i], x[:,L1[i][1]],
                x[:,L1[i][2]])
        end
        for i=1:length(L2)
            sqdistance(d2[i], x[:,L2[i][1]],
                x[:,L2[i][2]])
        end
        var_and_mean(v1, m1, d1)
        var_and_mean(v2, m2, d2)
    end
    out! += identity(v1)
    out! += identity(v2)
    ~@routine
end

function train(params)
    opt = Adam(lr=0.01)
    maxiter = 20000
    msk = [false, false, true, true, true, true, true, true, true, true]
    pp = params[:,msk]
    L1, L2 = petersen_bonds()
    for i=1:maxiter
        g = grad.(Grad(iloss)(Loss(0.0), params)[2][:,msk])
        update!(pp, g, opt)
        view(params, :, msk) .= pp
        if i%1000 == 0
            println("Step $i, loss = $(iloss(0.0, params)[1])")
        end
    end
    params
end

function petersen_bonds()
    L1 = [(1,6), (2,7), (3,8), (4,9), (5,10), (1,2), (2,3), (3,4), (4,5), (5,1), (6,8), (8,10), (10,7), (7,9), (9,6)]
    L1 = [i<j ? (i,j) : (j,i) for (i,j) in L1]
    LL = Any[]
    for i=1:9
        for j=i+1:10
            push!(LL, (i,j))
        end
    end
    L1, setdiff(LL, L1)
end

using BenchmarkTools
DO_BENCHMARK = false
if DO_BENCHMARK
    L1, L2 = petersen_bonds()
    x = randn(5,10)
    @benchmark Grad(iloss)(Loss(0.0), $x)
    @benchmark Zygote.gradient(loss, $x)
end

params = randn(4, 10)
@time train(params)
[Statistics.norm(params[:,i]-params[:,j]) for (i,j) in L1]
[Statistics.norm(params[:,i]-params[:,j]) for (i,j) in L2]

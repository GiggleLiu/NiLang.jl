using Zygote
import LinearAlgebra, Statistics

function myvar(v)
    mv  = Statistics.mean(v)
    sum((v .- mv).^2)./(length(v)-1)
end

function loss(x)
    L1 = [(1, 6), (2, 7), (3, 8), (4, 9), (5, 10),
    (1, 2), (2, 3), (3, 4), (4, 5), (1, 5), (6, 8),
    (8, 10), (7, 10), (7, 9), (6, 9)]
    L2 = [(1, 3), (1, 4), (1, 7), (1, 8), (1, 9),
    (1, 10), (2, 4), (2, 5), (2, 6), (2, 8), (2, 9),
    (2, 10), (3, 5), (3, 6), (3, 7), (3, 9), (3, 10),
    (4, 6), (4, 7), (4, 8), (4, 10), (5, 6), (5, 7),
    (5, 8), (5, 9), (6, 7), (6, 10), (7, 8), (8, 9),
    (9, 10)]
    a = [LinearAlgebra.norm(x[:,i]-x[:,j])^2 for (i, j) in L1]
    b = [LinearAlgebra.norm(x[:,i]-x[:,j])^2 for (i, j) in L2]
    myvar(a) + myvar(b)
end

function trainz(params)
    opt = Adam(lr=0.01)
    maxiter = 20000
    msk = [false, false, true, true, true, true, true, true, true, true]
    pp = params[:,msk]
    L1, L2 = petersen_bonds()
    for i=1:maxiter
        grad = view(Zygote.gradient(loss, params)[1], :,msk)
        update!(pp, grad, opt)
        view(params, :, msk) .= pp
        if i%1000 == 0
            println("Step $i, loss = $(loss(params))")
        end
    end
    params
end

Zygote.refresh()
params = randn(4, 10)
@time trainz(params)

L1 = [(1, 6), (2, 7), (3, 8), (4, 9), (5, 10),
(1, 2), (2, 3), (3, 4), (4, 5), (1, 5), (6, 8),
(8, 10), (7, 10), (7, 9), (6, 9)]
L2 = [(1, 3), (1, 4), (1, 7), (1, 8), (1, 9),
(1, 10), (2, 4), (2, 5), (2, 6), (2, 8), (2, 9),
(2, 10), (3, 5), (3, 6), (3, 7), (3, 9), (3, 10),
(4, 6), (4, 7), (4, 8), (4, 10), (5, 6), (5, 7),
(5, 8), (5, 9), (6, 7), (6, 10), (7, 8), (8, 9),
(9, 10)]
[Statistics.norm(params[:,i]-params[:,j]) for (i,j) in L1]
[Statistics.norm(params[:,i]-params[:,j]) for (i,j) in L2]

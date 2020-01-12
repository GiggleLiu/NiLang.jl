using NiLang, NiLang.AD
using LinearAlgebra

function _iqr(A)
    @assert size(A, 1) == size(A, 2)
    N = size(A, 1)
    Q = zero(A)
    R = zero(A)
    for col = 1:N
        ri = A[:,col]
        for precol = 1:col-1
            ni = Q[:,precol]'*ri
            R[precol,col] = ni
            ri = ri - ni*Q[:,precol]
        end
        R[col, col] = norm(ri)
        Q[:,col] = ri/R[col,col]
    end
    return Q, R
end

@i function iqr(Q, R, A::AbstractMatrix{T}) where T
    @anc anc_norm = zero(T)
    @anc anc_dot = zeros(T, size(A,2))
    @anc ri = zeros(T, size(A,1))
    for col = 1:size(A, 1)
        ri .+= identity.(A[:,col])
        for precol = 1:col-1
            idot(anc_dot[precol], Q[:,precol], ri)
            R[precol,col] += identity(anc_dot[precol])
            for row = 1:size(Q,1)
                ri[row] -= anc_dot[precol] * Q[row, precol]
            end
        end
        inorm2(anc_norm, ri)

        R[col, col] += anc_norm^0.5
        for row = 1:size(Q,1)
            Q[row,col] += ri[row] / R[col, col]
        end

        ~begin
            ri .+= identity.(A[:,col])
            for precol = 1:col-1
                idot(anc_dot[precol], Q[:,precol], ri)
                for row = 1:size(Q,1)
                    ri[row] -= anc_dot[precol] * Q[row, precol]
                end
            end
            inorm2(anc_norm, ri)
        end
    end
end

using Test, Random
@testset "test qr" begin
    Random.seed!(2)
    A = randn(4,4)
    Q, R = _iqr(A)
    @test Q * R ≈ A
    @test Q'*Q ≈ I
    R[abs.(R).<1e-10] .= 0.0
    @test istriu(R)

    q = zero(A)
    r = zero(A)
    @instr iqr(q, r, A)
    @test q ≈ Q
    @test r ≈ R
    @test check_inv(iqr, (q, r, A))

    @i function test1(out, q, r, A)
        iqr(q, r, A)
        out += identity(q[1,2])
    end
    @i function test2(out, q, r, A)
        iqr(q, r, A)
        out += identity(r[1,2])
    end
    @test check_grad(test1, (Loss(0.0), q, r, A); atol=0.05, verbose=true)
    @test check_grad(test2, (Loss(0.0), q, r, A); atol=0.05, verbose=true)
end

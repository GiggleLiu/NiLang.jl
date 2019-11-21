#=
export softmax_cross_entropy!
@invfunc softmax_cross_entropy!(x, p, imax, xmax, Z, out) begin
    @ancilla logZ::Float64
    @ancilla i::Int
    @ancilla N::Int
    # subtract maximum
    infer!(argmax, imax, x)  # trade off space of xmax to time
    add!(xmax, x[imax])
    sub!.(x, xmax)

    # accumulate exp(x) to Z, and finally get logZ
    infer!.(exp, Z, x)
    infer!(log, logZ, Z)
    sub!.(x, logZ)
    neg!.(x)
    ~infer!(log, logZ, Z)

    infer!(length, N, x)
    infer!.(*, out, x, p)
    ~infer!(length, N, x)
end
=#

export MapFoldl, umm!

struct MapFoldl{FM, FF} <: Function
    m::FM
    f::FF
    function MapFoldl(m::FM, f::FF) where {FM, FF}
        !isreversible(f) && throw(InvertibilityError("second parameter $f is not reversible!"))
        new{FM, FF}(m, f)
    end
end
#Base.:~(mf::MapFoldl) = MapFoldl(mf.m)
@i function (mf::MapFoldl)(out::T, iter) where T
    @anc anc::T
    for i=1:1:length(iter)
        anc += mf.m(iter[i])
        mf.f(out, anc)
        anc -= mf.m(iter[i])
    end
end

@i function umm!(x, θ, Nin::Int, Nout::Int)
    @anc k::Int
    for j=1:Nout
        for i=Nin-1:-1:j
            k ⊕ 1
            ROT(x[i], x[i+1], θ[k])
        end
    end

    for j=1:Nout
        for i=Nin-1:-1:j
            k ⊖ 1
        end
    end
end

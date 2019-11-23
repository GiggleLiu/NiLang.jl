export MapFoldl, umm!

struct MapFoldl{FM, FF} <: Function
    m::FM
    f::FF
    function MapFoldl(m::FM, f::FF) where {FM, FF}
        !isreversible(f) && throw(InvertibilityError("second parameter $f is not reversible!"))
        new{FM, FF}(m, f)
    end
end

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

    # uncompute k
    for j=1:Nout
        for i=Nin-1:-1:j
            k ⊖ 1
        end
    end
end

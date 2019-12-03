export MapFoldl, umm!
export inorm2, idot

struct MapFoldl{FM, FF} <: Function
    m::FM
    f::FF
    function MapFoldl(m::FM, f::FF) where {FM, FF}
        !isreversible(f) && throw(InvertibilityError("second parameter $f is not reversible!"))
        new{FM, FF}(m, f)
    end
end

@i function (mf::MapFoldl)(out::T, iter) where T
    @anc anc = zero(T)
    for i=1:1:length(iter)
        anc += mf.m(iter[i])
        mf.f(out, anc)
        anc -= mf.m(iter[i])
    end
end

# NOTE: also define the multiplication between two matrices
@i function umm!(x, θ, Nin::Int, Nout::Int)
    @anc k = 0
    for j=1:Nout
        for i=Nin-1:-1:j
            k += 1
            ROT(x[i], x[i+1], θ[k])
        end
    end

    # uncompute k
    for j=1:Nout
        for i=Nin-1:-1:j
            k -= 1
        end
    end
end

@i function idot(out, v1::AbstractVector{T}, v2) where T
    @anc anc1 = zero(T)
    for i = 1:length(v1)
        anc1 += v1[i]
        CONJ(anc1)
        out += v1[i]*v2[i]
        CONJ(anc1)
        anc1 -= v1[i]
    end
end

@i function inorm2(out, vec::AbstractVector{T}) where T
    @anc anc1 = zero(T)
    for i = 1:length(vec)
        anc1 += vec[i]
        CONJ(anc1)
        out += anc1*vec[i]
        CONJ(anc1)
        anc1 -= vec[i]
    end
end

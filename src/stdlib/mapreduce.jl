struct MapFoldl{FM, FF} <: Function
    m::FM
    f::FF
    function MapFoldl(m::FM, f::FF) where {FM, FF}
        !isreversible(f) && throw(InvertibilityError("second parameter $f is not reversible!"))
        new{FM, FF}(m, f)
    end
end

@i function (mf::MapFoldl)(out::T, iter) where T
    anc ← zero(T)
    for i=1:1:length(iter)
        anc += protectf(mf).m(iter[i])
        protectf(mf).f(out, anc)
        anc -= protectf(mf).m(iter[i])
    end
end

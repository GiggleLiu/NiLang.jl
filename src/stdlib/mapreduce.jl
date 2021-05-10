export i_mapfoldl, i_filter!, i_map!

"""
    i_mapfoldl(map, fold, out!, iter)

Reversible `mapfoldl` function, `map` can be irreversible, but `fold` should be reversible.
"""
@i function i_mapfoldl(map, fold, out!::T, iter) where T
    anc ← zero(T)
    for i=1:length(iter)
        anc += map(iter[i])
        fold(out!, anc)
        anc -= map(iter[i])
    end
    anc → zero(T)
end

"""
    i_filter!(f, out!, iter)

Reversible `filter` function, `out!` is an emptied vector.
"""
@i function i_filter!(f, out!::AbstractVector, x::AbstractVector{T}) where T
    @invcheckoff @inbounds for i = 1:length(x)
        if (f(x[i]), ~)
            COPYPUSH!(out!, x[i])
        end
    end
end

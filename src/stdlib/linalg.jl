export i_inv

"""
    i_inv(out!, A)

Get the inverse of `A`.

```note!!!
this function is implemented as a primitive.
```
"""
@i function i_inv(out!::AbstractMatrix{T}, A::AbstractMatrix{T}) where T
    @invcheckoff invA ← inv(A)
    out! .+= invA
    @invcheckoff invA → inv(A)
end

@i function i_inv(out!::AbstractMatrix{T}, A::AbstractMatrix{T}) where T<:GVar
    @routine @invcheckoff begin
        invA ← inv(value.(A))
        gA ← -transpose(invA) * grad(out!) * transpose(invA)
    end
    for i=1:length(out!)
        (out![i] |> value) -= invA[i]
    end
    for i=1:length(A)
        (A[i] |> grad) -= gA[i]
    end
    ~@routine
end

@i function ⊖(det)(out!::T, A::AbstractMatrix{T}) where T<:GVar
    @routine @invcheckoff begin
        vA ← value.(A)
        detA ← det(vA)
        gA ← detA * grad(out!) * transpose(inv(vA))
    end
    (out! |> value) -= detA
    for i=1:length(A)
        (A[i] |> grad) += gA[i]
    end
    ~@routine
end

@i function ⊖(logdet)(out!::T, A::AbstractMatrix{T}) where T<:GVar
    @routine @invcheckoff begin
        gA ← grad(out!) * transpose(inv(value.(A)))
    end
    (out! |> value) -= det(A |> grad)
    for i=1:length(A)
        (A[i] |> grad) += gA[i]
    end
    ~@routine
end

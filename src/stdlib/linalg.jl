export i_inv

@i function i_inv(out!::AbstractMatrix{T}, A::AbstractMatrix{T}) where T
    @invcheckoff invA ← inv(A)
    out! .+= identity.(invA)
    @invcheckoff invA → inv(A)
end

@i function i_inv(out!::AbstractMatrix{T}, A::AbstractMatrix{T}) where T<:GVar
    @routine @invcheckoff begin
        invA ← inv(value.(A))
        gA ← -transpose(invA) * grad(out!) * transpose(invA)
    end
    for i=1:length(out!)
        value(out![i]) -= identity(invA[i])
    end
    for i=1:length(A)
        grad(A[i]) -= identity(gA[i])
    end
    ~@routine
end

@i function ⊖(det)(out!::T, A::AbstractMatrix{T}) where T<:GVar
    @routine @invcheckoff begin
        vA ← value.(A)
        detA ← det(vA)
        gA ← detA * grad(out!) * transpose(inv(vA))
    end
    value(out!) -= identity(detA)
    for i=1:length(A)
        grad(A[i]) += identity(gA[i])
    end
    ~@routine
end

@i function ⊖(logdet)(out!::T, A::AbstractMatrix{T}) where T<:GVar
    @routine @invcheckoff begin
        gA ← grad(out!) * transpose(inv(value.(A)))
    end
    value(out!) -= det(grad(A))
    for i=1:length(A)
        grad(A[i]) += identity(gA[i])
    end
    ~@routine
end


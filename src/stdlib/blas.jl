export i_sum, i_mul!, i_dot, i_axpy!, i_umm!, i_norm2

@i function i_sum(out!, x::AbstractArray)
	@invcheckoff for i=1:length(x)
		@inbounds out! += x[i]
	end
end

@i function i_sum(out!, f, x::AbstractArray)
	@invcheckoff for i=1:length(x)
		@inbounds out! += f(x[i])
	end
end

@i function i_mul!(out!::AbstractMatrix{T}, x::AbstractMatrix{T}, y::AbstractMatrix{T}) where T
	@safe size(x, 2) == size(y, 1) || throw(DimensionMismatch())
	@invcheckoff @inbounds for k=1:size(y,2)
		for i=1:size(x,1)
			for j=1:size(x,2)
				out![i,k] += x[i,j] * y[j,k]
			end
		end
	end
end

@i function i_mul!(out!::AbstractVector{T}, x::AbstractMatrix, y::AbstractVector) where T
	@safe size(x, 2) == size(y, 1) || throw(DimensionMismatch())
	@invcheckoff @inbounds for j=1:size(x,2)
		yj ← zero(T)
		yj += y[j]
		for i=1:size(x,1)
			out![i] += x[i,j] * yj
		end
		yj -= y[j]
	end
end

@i function i_dot(out!, x, y)
    @safe @assert length(x) == length(y)
    @invcheckoff @inbounds for i=1:length(x)
        out! += x[i]' * y[i]
    end
end

@i function i_norm2(out!, x)
    @invcheckoff @inbounds for i=1:length(x)
        out! += abs2(x[i])
    end
end

@i function i_axpy!(a, X, Y)
    @safe @assert length(X) == length(Y)
    @invcheckoff @inbounds for i=1:length(Y)
        Y[i] += a * X[i]
    end
end

@i function i_umm!(x!::AbstractArray, θ)
    M ← size(x!, 1)
    N ← size(x!, 2)
    k ← 0
    @safe @assert length(θ) == M*(M-1)/2
    for l = 1:N
        for j=1:M
            for i=M-1:-1:j
                INC(k)
                ROT(x![i,l], x![i+1,l], θ[k])
            end
        end
    end

    k → length(θ)
end

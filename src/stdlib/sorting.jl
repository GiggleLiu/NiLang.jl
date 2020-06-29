export i_ascending!

"""
	i_ascending!(xs!, inds!, arr)

Rearrange `arr` into an ascending order `xs!`, indices are stored in `inds`.
"""
@i function i_ascending!(xs!::AbstractVector{T}, inds!, arr::AbstractArray{T}) where T
	@invcheckoff if (length(arr) > 0, ~)
		y ← zero(T)
		y += arr[1]
		ipush!(xs!, y)
		anc ← 1
		ipush!(inds!, anc)
		anc → 0
		@inbounds for i = 2:length(arr)
			if (arr[i] > xs![end], i==inds![end])
				ind ← i
				x ← zero(T)
				x += arr[i]
				ipush!(xs!, x)
				ipush!(inds!, ind)
				ind → 0
			end
		end
	end
end

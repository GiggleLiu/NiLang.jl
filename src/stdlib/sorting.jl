export i_ascending!

"""
	i_ascending!(xs!, inds!, arr)

Find the ascending sequence in `arr` and store the results into `xs!`, indices are stored in `inds!`.
This function can be used to get the maximum value and maximum indices.
"""
@i function i_ascending!(xs!::AbstractVector{T}, inds!, arr::AbstractArray{T}) where T
	@invcheckoff if (length(arr) > 0, ~)
		y ← zero(T)
		y += arr[1]
		xs![end+1] ↔ y
		anc ← 1
		inds![end+1] ↔ anc
		@inbounds for i = 2:length(arr)
			if (arr[i] > xs![end], i==inds![end])
				ind ← i
				x ← zero(T)
				x += arr[i]
				xs![end+1] ↔ x
				inds![end+1] ↔ ind
			end
		end
	end
end

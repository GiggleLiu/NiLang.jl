@i function (:-=)(gaussian_log)(y!::GVar{T}, x::GVar{T}) where T
	y!.x -= gaussian_log(x.x)
	@routine @invcheckoff begin
		exp_x ← zero(x)
		jac ← zero(x)
		exp_x += exp(-x)
	end
	x.g += y!.g * (exp_x |> AddConst(1) |> INV)
	~@routine
end

@i function (:-=)(gaussian_nlog)(y!::GVar{T}, x::GVar{T}) where T
	y!.x -= gaussian_nlog(x.x)
	@routine @invcheckoff begin
		exp_x ← zero(x)
		exp_x += exp(-x)
	end
	x.g -= y!.g * (exp_x |> SubConst(1) |> INV)
	~@routine
end

@i function :(-=)(convert)(out!::GVar{Tx, Tg}, y::ULogarithmic) where {Tx, Tg}
	out! -= exp(y.log)
end

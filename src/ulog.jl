using LogarithmicNumbers
export gaussian_log, gaussian_nlog
export ULogarithmic

function NiLangCore.default_constructor(ln::Type{<:ULogarithmic}, x)
	exp(ULogarithmic, x)
end

@i @inline function (:*=(identity))(x::ULogarithmic, y::ULogarithmic)
    x.log += y.log
end

@i @inline function (:*=(identity))(x::ULogarithmic, y::Real)
    x.log += log(y)
end

for (OP1, OP2, OP3) in [(:*, :+, :(+=)), (:/, :-, :(-=))]
	@eval @i @inline function (:*=($OP1))(out!::ULogarithmic, x::ULogarithmic, y::ULogarithmic)
	    out!.log += $OP2(x.log, y.log)
	end

	@eval @i @inline function (:*=($OP1))(out!::ULogarithmic, x::Real, y::Real)
	    out!.log += log(x)
		$(Expr(OP3, :(out!.log), :(log(y))))
	end

	@eval @i @inline function (:*=($OP1))(out!::ULogarithmic, x::ULogarithmic, y::Real)
	    out!.log += x.log
		$(Expr(OP3, :(out!.log), :(log(y))))
	end

	@eval @i @inline function (:*=($OP1))(out!::ULogarithmic, x::Real, y::ULogarithmic)
	    out!.log += log(x)
		$(Expr(OP3, :(out!.log), :(y.log)))
	end
end

@i @inline function (:*=(^))(out!::ULogarithmic, x::ULogarithmic, y::Real)
    out!.log += x.log * y
end

gaussian_log(x) = log1p(exp(x))
gaussian_nlog(x) = log1p(-exp(x))

@i function (:*=)(+)(out!::ULogarithmic{T}, x::ULogarithmic{T}, y::ULogarithmic{T}) where {T}
	@invcheckoff if (x.log == y.log, ~)
		out!.log += x.log
		out!.log += log(2)
	elseif (x.log ≥ y.log, ~)
		out!.log += x.log
		y.log -= x.log
		out!.log += gaussian_log(y.log)
		y.log += x.log
	else
		out!.log += y.log
		x.log -= y.log
		out!.log += gaussian_log(x.log)
		x.log += y.log
	end
end

@i function (:*=)(-)(out!::ULogarithmic{T}, x::ULogarithmic{T}, y::ULogarithmic{T}) where {T}
	@safe @assert x.log ≥ y.log
	@invcheckoff if (!iszero(x), ~)
		out!.log += x.log
		y.log -= x.log
		out!.log += gaussian_nlog(y.log)
		y.log += x.log
	end
end

@i function :(*=)(convert)(out!::ULogarithmic{T}, y::ULogarithmic) where T
    out!.log += convert((@skip! T), y.log)
end

@i function :(*=)(convert)(out!::ULogarithmic{T}, y::T) where T<:Real
    out!.log += log(y)
end

Base.convert(::Type{T}, x::ULogarithmic{T}) where {T<:Fixed} = exp(x.log)

function NiLangCore.deanc(x::T, v::T) where T<:ULogarithmic
    x === v || NiLangCore.deanc(x.log, v.log)
end

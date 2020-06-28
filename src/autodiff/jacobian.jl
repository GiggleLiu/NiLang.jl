export jacobian, jacobian_repeat

"""
    jacobian_repeat(f, args...; iin::Int, iout::Int=iin, kwargs...)

Get the Jacobian matrix for function `f(args..., kwargs...)` using repeated computing gradients for each output.
One can use key word arguments `iin` and `iout` to specify the input and output tensor.
"""
function jacobian_repeat(f, args...; iin::Int, iout::Int=iin, kwargs...)
    _check_input(args, iin, iout)
    N = length(args[iout])
    res = zeros(eltype(args[iin]), length(args[iin]), N)
    for i = 1:N
        xargs = _copy.(args)
        xargs = NiLangCore.wrap_tuple(f(xargs...; kwargs...))
        xargs = GVar.(xargs)
        @inbounds xargs[iout][i] = GVar(value(xargs[iout][i]), one(eltype(args[iout])))
        @inbounds res[:,i] .= vec(grad.(NiLangCore.wrap_tuple((~f)(xargs...; kwargs...))[iin]))
    end
    return res
end

_copy(x) = x
_copy(x::AbstractArray) = copy(x)

"""
    jacobian(f, args...; iin::Int, iout::Int=iin, kwargs...)

Get the Jacobian matrix for function `f(args..., kwargs...)` using vectorized variables in the gradient field.
One can use key word arguments `iin` and `iout` to specify the input and output tensor.
"""
function jacobian(f, args...; iin::Int, iout::Int=iin, kwargs...)
    _check_input(args, iin, iout)
    args = NiLangCore.wrap_tuple(f(args...; kwargs...))
    ABT = AutoBcast{eltype(args[iout]), length(args[iout])}
    _args = map(i-> i==iout ? wrap_jacobian(ABT, args[i]) : wrap_bcastgrad(ABT, args[i]), 1:length(args))
    _args = NiLangCore.wrap_tuple((~f)(_args...; kwargs...))
    out = zeros(eltype(args[iin]), length(args[iin]), length(args[iout]))
    for i=1:length(args[iin])
        @inbounds out[i,:] .= grad(_args[iin][i]).x
    end
    out
end

function wrap_jacobian(::Type{AutoBcast{T,N}}, outarray::AbstractArray{T}) where {T,N}
    map(k->GVar(outarray[k], AutoBcast{T,N}(onehot(T, N, k))), LinearIndices(outarray))
end
function wrap_bcastgrad(::Type{AutoBcast{T,N}}, x::XT) where {T,N,XT}
    GVar(x, zero(AutoBcast{XT,N}))
end
function wrap_bcastgrad(::Type{AutoBcast{T,N}}, x::Union{Integer, Function}) where {T,N,XT}
    x
end
function wrap_bcastgrad(::Type{AutoBcast{T,N}}, x::NoGrad) where {T,N,XT}
    (~NoGrad)(x)
end
function wrap_bcastgrad(::Type{AutoBcast{T,N}}, x::Union{Tuple,AbstractArray}) where {T,N,XT}
    wrap_bcastgrad.(AutoBcast{T,N}, x)
end

function onehot(::Type{T}, N::Int, k::Int) where T
    res = zeros(T, N)
    res[k] = one(T)
    res
end

function _check_input(args, iin, iout)
    if !(args[iin] isa AbstractArray && args[iout] isa AbstractArray)
        throw(ArgumentError("argument at position $iin and $iout are not arrays."))
    elseif (eltype(args[iin]) != eltype(args[iout]))
        throw(ArgumentError("argument at position $iin and $iout do not have the same type."))
    end
end

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
        xargs = copy.(args)
        xargs = NiLangCore.wrap_tuple(f(xargs...; kwargs...))
        xargs = GVar.(xargs)
        xargs[iout][i] = GVar(value(xargs[iout][i]), one(eltype(args[iout])))
        res[:,i] .= grad.(NiLangCore.wrap_tuple((~f)(xargs...; kwargs...))[iin])
    end
    return res
end

"""
    jacobian(f, args...; iin::Int, iout::Int=iin, kwargs...)

Get the Jacobian matrix for function `f(args..., kwargs...)` using vectorized variables in the gradient field.
One can use key word arguments `iin` and `iout` to specify the input and output tensor.
"""
function jacobian(f, args...; iin::Int, iout::Int=iin, kwargs...)
    _check_input(args, iin, iout)
    args = NiLangCore.wrap_tuple(f(args...; kwargs...))
    _args = map(i-> i==iout ? wrap_jacobian(args[i]) : GVar(args[i]), 1:length(args))
    _args = NiLangCore.wrap_tuple((~f)(_args...; kwargs...))
    out = zeros(eltype(args[iin]), length(args[iin]), length(args[iout]))
    for i=1:length(args[iin])
        @inbounds out[i,:] .= grad(_args[iin][i]).x
    end
    out
end

function wrap_jacobian(outarray::AbstractArray{T}) where T
    N = length(outarray)
    map(k->GVar(outarray[k], AutoBcast(onehot(T, N, k))), 1:N)
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

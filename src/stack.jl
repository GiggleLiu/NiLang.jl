export POP!, PUSH!, COPYPUSH!, COPYPOP!, GLOBAL_STACK

const GLOBAL_STACK = []

############# global stack operations ##########
@inline function PUSH!(x)
    push!(GLOBAL_STACK, x)
    _zero(x)
end

@inline function POP!(x::T) where T
    NiLangCore.deanc(x, _zero(x))
    loaddata(T, pop!(GLOBAL_STACK))
end

UNSAFE_PUSH!(args...) = PUSH!(args...)
@inline function UNSAFE_POP!(x::T) where T
    loaddata(T, pop!(GLOBAL_STACK))
end

############# local stack operations ##########
@inline function PUSH!(stack, x::T) where T
    push!(stack, x)
    stack, _zero(T)
end

@inline function POP!(stack, x::T) where T
    NiLangCore.deanc(x, _zero(T))
    stack, loaddata(T, pop!(stack))
end

@inline function UNSAFE_POP!(stack, x::T) where T
    stack, loaddata(T, pop!(stack))
end

loaddata(::Type{T}, x::T) where T = x

@dual POP! PUSH!
@dual UNSAFE_POP! UNSAFE_PUSH!

############# copied push/pop stack operations ##########
@inline function COPYPUSH!(stack, x)
    push!(stack, copy(x))
    stack, x
end

@inline function COPYPOP!(stack, x)
    y = pop!(stack)
    NiLangCore.deanc(x, y)
    stack, y
end

UNSAFE_COPYPUSH!(args...) = COPYPUSH!(args...)
@inline function UNSAFE_COPYPOP!(stack, x)
    y = pop!(stack)
    stack, y
end

@inline function COPYPUSH!(x)
    push!(GLOBAL_STACK, x)
    x
end

@inline function COPYPOP!(x)
    y = pop!(GLOBAL_STACK)
    NiLangCore.deanc(x, y)
    x
end

@inline function UNSAFE_COPYPOP!(x)
    y = pop!(GLOBAL_STACK)
    y
end

@dual COPYPOP! COPYPUSH!
@dual UNSAFE_COPYPOP! UNSAFE_COPYPUSH!

loaddata(::Type{T}, x::TX) where {T<:NullType, TX} = T(x)

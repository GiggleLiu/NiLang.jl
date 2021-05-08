export POP!, PUSH!, COPYPUSH!, COPYPOP!, GLOBAL_STACK

const GLOBAL_STACK = []

struct Stack{T}
    data::Vector{T}
    top::Base.RefValue{Int}
end

function Stack{T}(n::Int)
    Stack{T}(zeros(T, n), Ref(0))
end

@inline function Base.push!(stack::Stack, val)
    stack.top[] += 1
    @boundscheck stack.top[] <= length(stack.data) || throw(BoundsError("stack of size `$(length(stack))` is too small to fit your data"))
    stack.data[stack.top[]] = val
    return stack
end

@inline function Base.pop!(stack::Stack)
    @boundscheck stack.top[] <= 0 || throw(BoundsError("hit stack bottom"))
    val = stack.data[stack.top[]]
    stack.top[] -= 1
    return val
end

# default stack size is 10^6 (~8M for Float64)
for DT in [:Float64, :Float32, :ComplexF64, :ComplexF32, :Int64, :Int32, :Bool, :UInt8]
    STACK = Symbol(:GLOBAL_STACK_, DT)
    @eval const $STACK = Stack{$DT}(1000000)
    @eval @inline function PUSH!(x::$DT)
        push!($STACK, x)
        _zero($DT)
    end
    @eval @inline function POP!(x::$DT)
        NiLangCore.deanc(x, _zero($DT))
        loaddata($DT, pop!($STACK))
    end
end

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
@inline function UNSAFE_COPYPOP!(stack, x::XT) where XT
    y = pop!(stack)
    stack, convert(XT, y)
end

@inline function COPYPUSH!(x)
    push!(GLOBAL_STACK, x)
    x
end

@inline function COPYPOP!(x::XT) where XT
    y = pop!(GLOBAL_STACK)
    NiLangCore.deanc(x, y)
    convert(XT, y)
end

@inline function UNSAFE_COPYPOP!(x)
    y = pop!(GLOBAL_STACK)
    y
end

@dual COPYPOP! COPYPUSH!
@dual UNSAFE_COPYPOP! UNSAFE_COPYPUSH!

loaddata(::Type{T}, x::TX) where {T<:NullType, TX} = T(x)
